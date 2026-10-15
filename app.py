"""Exercise 5 — Streamlit approval UI for the HITL PR review agent.

Run with:
    uv run streamlit run app.py

Goal: wrap the LangGraph built in exercises 1–4 in a web UI that adapts to
the confidence bucket of each PR.

Routing thresholds (common/schemas.py):
    > 72%        auto_approve     UI shows a success card; reviewer does nothing
    58 – 72%     human_approval   UI shows Approve / Reject / Edit buttons
    <  58%       escalate         UI shows a question form for the reviewer,
                                  then the refined review for a final confirm
"""

from __future__ import annotations

import asyncio
import uuid

import streamlit as st
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from audit.replay import fetch_threads
from common.db import db_path
from exercises.exercise_4_audit import build_graph


load_dotenv()

RISK_ICON = {"low": ":material/check_circle:", "med": ":material/help:", "high": ":material/warning:"}


# ─── Session state ─────────────────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "pr_url" not in st.session_state:
    st.session_state.pr_url = ""
if "interrupt_payload" not in st.session_state:
    st.session_state.interrupt_payload = None
if "final" not in st.session_state:
    st.session_state.final = None


# ─── Drive the graph ───────────────────────────────────────────────────────
async def run_graph(pr_url: str, thread_id: str, resume_value=None):
    """Invoke the graph once. Returns the final result or {'__interrupt__': ...}."""
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp, confirm_after_escalation=True)
        cfg = {"configurable": {"thread_id": thread_id}}
        if resume_value is None:
            return await app.ainvoke({"pr_url": pr_url, "thread_id": thread_id}, cfg)
        return await app.ainvoke(Command(resume=resume_value), cfg)


async def load_thread(thread_id: str):
    """Read the checkpointed state of an existing thread (no LLM / GitHub calls)."""
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        return await build_graph(cp, confirm_after_escalation=True).aget_state(
            {"configurable": {"thread_id": thread_id}}
        )


def apply_result(result: dict) -> None:
    if "__interrupt__" in result:
        st.session_state.interrupt_payload = result["__interrupt__"][0].value
        st.session_state.final = None
    else:
        st.session_state.interrupt_payload = None
        st.session_state.final = result


def open_thread(thread_id: str) -> None:
    """Restore a session (sidebar click or ?thread=<id> after a page reload)."""
    snap = asyncio.run(load_thread(thread_id))
    values = snap.values or {}
    interrupts = [i for t in snap.tasks for i in t.interrupts]
    st.session_state.thread_id = thread_id
    st.session_state.pr_url = values.get("pr_url", "")
    st.session_state.interrupt_payload = interrupts[0].value if interrupts else None
    st.session_state.final = values if not interrupts and values.get("final_action") else None
    st.query_params["thread"] = thread_id


# Reloading the page drops session_state; the thread id survives in the URL.
if st.session_state.thread_id is None and "thread" in st.query_params:
    open_thread(st.query_params["thread"])


# ─── Page setup ────────────────────────────────────────────────────────────
st.set_page_config(page_title="HITL PR review", layout="wide")
st.title("HITL PR review agent")


# ─── Sidebar — recent sessions ─────────────────────────────────────────────
with st.sidebar:
    st.header("Recent sessions")
    threads = asyncio.run(fetch_threads())
    if not threads:
        st.caption("No sessions yet.")
    for t in threads:
        pr = t["pr_url"].split("github.com/")[-1]
        st.button(
            f"{pr} · {t['thread_id'][:8]}",
            key=f"open_{t['thread_id']}",
            icon=RISK_ICON.get(t["worst_risk"]),
            help=f"worst risk: {t['worst_risk']} · {t['events']} events · last {t['last_event'][:19]}",
            type="primary" if t["thread_id"] == st.session_state.thread_id else "secondary",
            width="stretch",
            on_click=open_thread,
            args=(t["thread_id"],),
        )


# ─── Top form — start a new review ─────────────────────────────────────────
with st.form("start"):
    pr_url = st.text_input(
        "PR URL", value=st.session_state.pr_url,
        placeholder="https://github.com/VinUni-AI20k/PR-Demo/pull/1",
    )
    submitted = st.form_submit_button("Run review", icon=":material/play_arrow:")


# ─── Renderers per interrupt kind ──────────────────────────────────────────
def render_approval_card(payload: dict) -> dict | None:
    """58–72% bucket: show the LLM review + 3 buttons. Return resume dict or None."""
    conf = payload["confidence"]
    st.subheader(f"Approval requested — confidence {conf:.0%}")
    st.caption(payload["confidence_reasoning"])
    st.markdown(payload["summary"])

    for c in payload.get("comments", []):
        st.markdown(f"- **[{c['severity']}]** `{c['file']}:{c.get('line') or '?'}` — {c['body']}")

    with st.expander("Diff"):
        st.code(payload.get("diff_preview", ""), language="diff")

    feedback = st.text_input("Feedback (optional)", key=f"feedback_{st.session_state.thread_id}")
    row = st.container(horizontal=True)
    if row.button("Approve", type="primary", icon=":material/check:"):
        return {"choice": "approve", "feedback": feedback}
    if row.button("Reject", icon=":material/close:"):
        return {"choice": "reject", "feedback": feedback}
    if row.button("Edit", icon=":material/edit:"):
        return {"choice": "edit", "feedback": feedback}
    return None


def render_escalation_card(payload: dict) -> dict | None:
    """< 58% bucket: show risk factors + question form. Return {question: answer} or None."""
    conf = payload["confidence"]
    st.subheader(f"Strong escalation — confidence {conf:.0%}")
    st.caption(payload["confidence_reasoning"])
    if payload.get("risk_factors"):
        st.error("\n".join(f"- {r}" for r in payload["risk_factors"]), title="Risk factors")
    st.markdown(payload["summary"])

    with st.form("escalation"):
        answers = {
            q: st.text_area(q, key=f"q{i}_{st.session_state.thread_id}")
            for i, q in enumerate(payload["questions"])
        }
        if st.form_submit_button("Submit answers", type="primary"):
            if all(a.strip() for a in answers.values()):
                return {q: a.strip() for q, a in answers.items()}
            st.warning("Please answer every question.")
    return None


# ─── Main flow ─────────────────────────────────────────────────────────────
if submitted and pr_url:
    st.session_state.pr_url = pr_url
    st.session_state.thread_id = str(uuid.uuid4())
    st.query_params["thread"] = st.session_state.thread_id

    with st.spinner("Fetching PR + asking the LLM..."):
        apply_result(asyncio.run(run_graph(pr_url, st.session_state.thread_id)))
    st.rerun()  # refresh the sidebar with the new session

# Render the current interrupt card, if any
payload = st.session_state.interrupt_payload
if payload is not None:
    kind = payload["kind"]
    answer = render_approval_card(payload) if kind == "approval_request" else render_escalation_card(payload)
    if answer is not None:
        with st.spinner("Resuming..."):
            apply_result(asyncio.run(run_graph(
                st.session_state.pr_url, st.session_state.thread_id, resume_value=answer,
            )))
        st.rerun()

# Render final state, if reached
if st.session_state.final is not None:
    final = st.session_state.final
    action = final.get("final_action", "?")
    a = final.get("analysis")
    if action.endswith("failed"):
        st.error(f"{action} — could not post the comment (check GITHUB_TOKEN scope)")
    elif action.startswith("auto"):
        with st.container(border=True):
            st.success("High confidence — the agent posted the review without a human.",
                       title=f"Auto-approved · confidence {a.confidence:.0%}")
            st.markdown(a.summary)
            st.link_button("View comment on GitHub", st.session_state.pr_url, icon=":material/open_in_new:")
    elif action.startswith("committed"):
        st.success(f"{action} — comment posted to {st.session_state.pr_url}")
        st.link_button("View comment on GitHub", st.session_state.pr_url, icon=":material/open_in_new:")
    elif action == "rejected":
        st.warning(f"Not posted (reviewer chose {final.get('human_choice')})")
    else:
        st.info(f"final_action = {action}")

if st.session_state.thread_id:
    st.caption(f"thread_id = {st.session_state.thread_id}  ·  replay: "
               f"`uv run python -m audit.replay --thread {st.session_state.thread_id}`")
