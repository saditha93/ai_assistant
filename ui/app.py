import json
import os
import uuid

import httpx
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Crestline Assistant", layout="wide")
state = st.session_state
state.setdefault("token", None)
state.setdefault("user", None)
state.setdefault("session_id", uuid.uuid4().hex[:12])
state.setdefault("messages", [])
state.setdefault("activity", [])
state.setdefault("pending_approval", None)
state.setdefault("feedback_sent", set())


def headers() -> dict:
    return {"Authorization": f"Bearer {state.token}"}


def api(method: str, path: str, **kwargs):
    try:
        r = httpx.request(method, f"{API_URL}{path}", headers=headers(), timeout=30, **kwargs)
        return r.json() if r.status_code < 400 else None
    except httpx.HTTPError:
        return None


def stream_events(path: str, body: dict):
    """Yield (event, data) pairs from the backend's Server-Sent Events."""
    try:
        with httpx.stream("POST", f"{API_URL}{path}", json=body, headers=headers(), timeout=300) as r:
            if r.status_code != 200:
                r.read()
                detail = r.json().get("detail") if "json" in r.headers.get("content-type", "") else r.text
                yield "http_error", {"status": r.status_code, "detail": detail}
                return
            event = None
            for line in r.iter_lines():
                if line.startswith("event: "):
                    event = line[7:]
                elif line.startswith("data: ") and event:
                    yield event, json.loads(line[6:])
    except httpx.HTTPError as exc:
        yield "http_error", {"status": 0, "detail": f"Cannot reach the API at {API_URL} ({exc})"}


# ---- activity panel ----

def render_event(box, e: dict) -> None:
    kind = e.get("type")
    if kind == "node":
        if e["status"] == "start":
            box.markdown(f":blue[**{e['node']}**] started")
        elif e["status"] == "end":
            box.caption(f"{e['node']} finished in {e.get('ms', 0)} ms")
        elif e["status"] == "error":
            box.markdown(f":red[**{e['node']} failed**] {e.get('error')} - continuing in degraded mode")
    elif kind == "guard":
        verdict = "blocked" if not e["allowed"] else ("flagged" if e["flagged"] else "passed")
        colour = {"blocked": "red", "flagged": "orange", "passed": "green"}[verdict]
        labels = ", ".join(e["labels"]) or "none"
        box.markdown(f":{colour}[Input guard: **{verdict}**] risk {e['score']} (signals: {labels})")
    elif kind == "memory":
        if e["status"] == "loaded":
            box.markdown(f"Memory loaded: {e['session_turns']} earlier turns in this session, "
                         f"summary: {'yes' if e['has_summary'] else 'no'}, facts: {e['facts'] or 'none'}")
            if e["related_questions"]:
                box.caption("Related past questions: " + " | ".join(e["related_questions"]))
        else:
            box.markdown(f"Memory saved: interaction stored, {e['summarised_messages']} old messages summarised, "
                         f"new facts: {e['new_facts'] or 'none'}")
    elif kind == "plan":
        box.markdown(f"**Plan** intent `{e['intent']}` - steps: `{' -> '.join(e['steps']) or 'answer directly'}`")
        box.caption(f"Standalone question: {e['question']}")
        box.caption(f"Sub-tasks: {'; '.join(e['sub_tasks'])} | namespaces: {', '.join(e['namespaces'])} | "
                    f"filters: {e['filters'] or 'none'}")
        box.caption(f"Reasoning: {e['reasoning']}")
        for note in e.get("notes", []):
            box.markdown(f":orange[{note}]")
    elif kind == "retrieval":
        if e["status"] == "searching":
            box.markdown(f"Retrieval: searching `{e['query']}` in {', '.join(e['namespaces'])} "
                         f"(filters: {e.get('filters') or 'none'})")
        elif e["status"] == "retrying":
            box.markdown(f":orange[Retrieval retry] {e['reason']} - new query `{e['query']}`")
        else:
            box.markdown(f"Retrieval done: mode **{e['mode']}**, reranked: {e['reranked']}, {len(e['hits'])} hits")
            rows = [{"id": h["id"], "title": h["title"][:40], "dense": h["dense_score"], "sparse": h["sparse_score"],
                     "hybrid": h["score"], "rerank": h.get("rerank_score")} for h in e["hits"]]
            if rows:
                box.dataframe(rows, hide_index=True, width="stretch")
            for note in e["notes"]:
                box.markdown(f":orange[{note}]")
            for flag in e["injection_flags"]:
                box.markdown(f":red[Injection removed from {flag['chunk']}]: {', '.join(flag['labels'])}")
    elif kind == "research":
        status = e["status"]
        if status == "start":
            box.markdown(f"**Research (depth {e['depth']})** on {e['documents']} documents: {e['task']}")
        elif status == "cell":
            box.caption(f"Depth {e['depth']}, step {e['cell']}: generated search plan")
            box.code(e["code"], language="python")
        elif status == "cell_result":
            box.caption(("Output" if e["ok"] else "Error") + f" of step {e['cell']}:")
            box.text(e["output"])
        elif status == "sub_agent":
            box.markdown(f":violet[Recursive sub-agent (depth {e['depth']})] {e['task']} on {e['doc_ids']}")
        elif status in ("sub_query", "sub_query_batch"):
            box.caption(f"Sub-LLM call: {e.get('prompt') or str(e.get('count')) + ' prompts in parallel'}")
        elif status == "fixed_plan":
            box.markdown(f":orange[Fixed research plan] over {e['documents']}")
        elif status == "batch":
            box.caption(f"Batch {e['batch']}: {', '.join(e['doc_ids'])}")
        elif status == "fallback":
            box.markdown(f":orange[Research loop fell back to fixed plan]: {e['reason']}")
        elif status == "final":
            box.caption(f"Research depth {e['depth']} finished")
    elif kind == "tool":
        if e["status"] == "call":
            box.markdown(f"Tool call **{e['tool']}** `{json.dumps(e['args'])[:200]}`")
        elif e["status"] == "denied":
            box.markdown(f":red[Tool denied] {e['tool']}: {e['reason']}")
        elif e["status"] == "result":
            box.caption(f"{e['tool']} returned in {e['ms']} ms: {e['preview']}")
        else:
            box.markdown(f":red[Tool error] {e['tool']}: {e['error']}")
    elif kind == "approval":
        box.markdown(f":orange[Approval {e['status']}] {e.get('tool')}")
    elif kind == "validation":
        colour = "green" if e["passed"] else "red"
        detail = f" {e['detail']}" if e.get("detail") else ""
        box.markdown(f":{colour}[{'pass' if e['passed'] else 'FAIL'}] {e['check']}{detail}")
    elif kind == "answer_reset":
        box.markdown(f":orange[Draft rejected ({', '.join(e['reason'])}) - regenerating]")
    elif kind == "response":
        box.markdown(f"Response: **{e['status']}**" + (f" ({e['reason']})" if e.get("reason") else ""))


# ---- chat ----

def run_turn(path: str, body: dict, chat_box, status_line) -> None:
    state.activity = []
    activity_box = new_activity_box()
    with chat_box.chat_message("assistant"):
        placeholder = st.empty()
        text, final = "", None
        for event, data in stream_events(path, body):
            if event == "http_error":
                (st.warning if data["status"] == 429 else st.error)(data["detail"])
                return
            if event == "activity":
                state.activity.append(data)
                render_event(activity_box, data)
                if data.get("type") == "node" and data["status"] == "start":
                    status_line.info(f"Active node: **{data['node']}**")
                if data.get("type") == "answer_reset":
                    text = ""
                    placeholder.markdown("_Revising the answer..._")
            elif event == "token":
                text += data
                placeholder.markdown(text + " ▌")
            elif event == "approval_required":
                state.pending_approval = data
            elif event == "answer":
                final = data
                placeholder.markdown(data["answer"])
            elif event == "error":
                st.error(data["message"])
        status_line.success("Idle")
    if final:
        state.messages.append({"role": "assistant", "content": final["answer"], "citations": final["citations"],
                               "run_id": final["run_id"], "notes": final["notes"]})
    elif state.pending_approval and text:
        state.messages.append({"role": "assistant", "content": text})


def render_message(m: dict, i: int) -> None:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("citations"):
            with st.expander(f"Sources ({len(m['citations'])})"):
                for c in m["citations"]:
                    st.markdown(f"- `{c['id']}` {c['title']} - {c['section']}")
        if m.get("run_id"):
            st.caption(f"LangSmith run id: {m['run_id']}")
            score = st.feedback("thumbs", key=f"fb_{i}")
            if score is not None and m["run_id"] not in state.feedback_sent:
                api("POST", "/feedback", json={"run_id": m["run_id"], "score": score})
                state.feedback_sent.add(m["run_id"])
                st.toast("Thanks for the feedback")


# ---- sidebar ----

with st.sidebar:
    st.header("Crestline Assistant")
    if not state.token:
        with st.form("login"):
            username = st.text_input("Username", value="analyst1")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Sign in"):
                try:
                    r = httpx.post(f"{API_URL}/auth/login", json={"username": username, "password": password})
                except httpx.HTTPError:
                    r = None
                if r is not None and r.status_code == 200:
                    state.token, state.user = r.json()["token"], r.json()["user"]
                    st.rerun()
                st.error("Sign-in failed" if r is not None else f"API not reachable at {API_URL}")
        st.caption("Demo users: viewer1 / viewer123, analyst1 / analyst123, admin1 / admin123")
        st.stop()

    user = state.user
    st.markdown(f"**{user['name']}**  \nRole: `{user['role']}` - {user['department']}")
    st.caption("Tools: " + ", ".join(user["tools"]))
    st.caption(f"Session: {state.session_id}")
    if st.button("New conversation"):
        state.session_id, state.messages, state.activity, state.pending_approval = uuid.uuid4().hex[:12], [], [], None
        st.rerun()
    if st.button("Sign out"):
        state.clear()
        st.rerun()

    health = api("GET", "/health")
    if health:
        with st.expander("System health", expanded=False):
            for dep, value in health["dependencies"].items():
                st.markdown(f"- {dep}: `{value}`")
            st.markdown("Circuit breakers: " + ", ".join(f"{k} `{v}`" for k, v in health["circuit_breakers"].items()))

    if user["role"] == "admin":
        st.subheader("Fault injection")
        st.caption("Simulate failures to show graceful degradation.")
        faults = api("GET", "/admin/faults") or {}
        changes = {name: st.toggle(name, value=on, key=f"fault_{name}") for name, on in faults.items()}
        if changes and changes != faults:
            api("POST", "/admin/faults", json=changes)
            st.rerun()
        with st.expander("Audit log"):
            for entry in api("GET", "/admin/audit", params={"limit": 20}) or []:
                st.caption(f"{entry['ts']} {entry['user']} {entry['event']} {json.dumps(entry['detail'])[:120]}")


# ---- main layout ----

chat_col, activity_col = st.columns([3, 2], gap="large")
with activity_col:
    st.subheader("Agent activity")
    status_line = st.empty()
    status_line.success("Idle")
    activity_slot = st.empty()

    def new_activity_box():
        return activity_slot.container(height=720, autoscroll=True)

    activity_box = new_activity_box()
    for event in state.activity:
        render_event(activity_box, event)

with chat_col:
    st.subheader("Chat")
    chat_box = st.container(height=640, autoscroll=True)
    for i, m in enumerate(state.messages):
        with chat_box:
            render_message(m, i)

    if state.pending_approval:
        action = state.pending_approval
        with chat_box.container(border=True):
            st.markdown(f"**Approval needed:** the assistant wants to run `{action['tool']}`")
            st.json(action["args"])
            approve, reject = st.columns(2)
            decision = None
            if approve.button("Approve", type="primary"):
                decision = True
            if reject.button("Reject"):
                decision = False
        if decision is not None:
            state.pending_approval = None
            run_turn("/chat/resume", {"session_id": state.session_id, "approved": decision}, chat_box, status_line)
            st.rerun()

    question = st.chat_input("Ask about policies, runbooks, incidents, architecture...",
                             disabled=bool(state.pending_approval))
    if question:
        state.messages.append({"role": "user", "content": question})
        with chat_box:
            render_message(state.messages[-1], len(state.messages) - 1)
        run_turn("/chat/stream", {"message": question, "session_id": state.session_id}, chat_box, status_line)
        st.rerun()
