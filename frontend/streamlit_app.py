"""Optional Streamlit frontend client communicating with the FastAPI backend."""

import os
import requests
import streamlit as st

BACKEND_URL = os.getenv("WORKBENCH_API_URL", "http://127.0.0.1:8000/api/v1")

st.set_page_config(
    page_title="Sovereign On-Premise AI Workbench",
    layout="wide",
)

st.title("Sovereign On-Premise Agentic AI Workbench")
st.caption("Local-first RAG, bounded agent orchestration, and multimodal vision analysis")

with st.sidebar:
    st.subheader("System Status")
    try:
        status_resp = requests.get(f"{BACKEND_URL}/system/status", timeout=5)
        if status_resp.ok:
            status = status_resp.json()
            st.success(f"Running ({status.get('environment')})")
            st.json(status.get("models", {}))
            st.json(status.get("components", {}))
        else:
            st.error("Backend returned an error.")
    except requests.RequestException:
        st.error("FastAPI backend is unreachable at " + BACKEND_URL)

tab_qa, tab_agent, tab_vision, tab_docs, tab_models = st.tabs(
    ["Document QA", "Bounded Agent", "Vision Analysis", "Documents", "Models & Security"]
)

with tab_qa:
    docs = []
    try:
        r = requests.get(f"{BACKEND_URL}/documents/", timeout=5)
        if r.ok:
            docs = r.json().get("documents", [])
    except requests.RequestException:
        pass

    doc_options = {"All Approved Documents": None}
    for d in docs:
        doc_options[f"{d['filename']} ({d['document_id'][:8]})"] = d["document_id"]

    selected_label = st.selectbox("Document scope", list(doc_options.keys()))
    retrieval_mode = st.radio("Retrieval mode", ["semantic", "hybrid"], horizontal=True)
    question = st.text_area("Ask a question about uploaded documents")

    if st.button("Ask Local RAG") and question.strip():
        resp = requests.post(
            f"{BACKEND_URL}/qa/ask",
            json={
                "question": question.strip(),
                "document_id": doc_options[selected_label],
                "retrieval_mode": retrieval_mode,
            },
            timeout=120,
        )
        if resp.ok:
            data = resp.json()
            st.markdown(data["answer"])
            st.caption(
                f"Method: {data.get('answer_method')} | Latency: {data.get('latency_seconds')}s"
            )
            if data.get("claim_verification"):
                st.json(data["claim_verification"])
            with st.expander("Sources & Page References"):
                for src in data.get("sources", []):
                    st.markdown(
                        f"**{src['filename']}** — Page {src['page_number']} "
                        f"(Chunk {src['chunk_index']}, Score: {src.get('similarity_score')})"
                    )
                    st.text(src["text"])
        else:
            st.error(resp.text)

with tab_agent:
    agent_q = st.text_area("Agent request or calculation")
    agent_img = st.file_uploader("Optional image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if st.button("Run Bounded Agent") and agent_q.strip():
        if agent_img is not None:
            resp = requests.post(
                f"{BACKEND_URL}/agent/run-with-image",
                data={"question": agent_q.strip()},
                files={"image": (agent_img.name, agent_img.getvalue(), agent_img.type)},
                timeout=180,
            )
        else:
            resp = requests.post(
                f"{BACKEND_URL}/agent/run",
                json={"question": agent_q.strip()},
                timeout=180,
            )
        if resp.ok:
            data = resp.json()
            st.markdown(data["answer"])
            st.caption(
                f"Status: {data.get('termination_reason')} | Plan: {data.get('plan_source')}"
            )
            with st.expander("Execution Trace"):
                st.json(data.get("trace", []))
        else:
            st.error(resp.text)

with tab_vision:
    v_task = st.selectbox("Task", ["scene", "vqa", "document"])
    v_question = st.text_input("Question (required for VQA)")
    v_img = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"], key="vision_upload")
    if st.button("Analyze Image") and v_img is not None:
        form = {"task": v_task}
        if v_question.strip():
            form["question"] = v_question.strip()
        resp = requests.post(
            f"{BACKEND_URL}/vision/analyze",
            data=form,
            files={"file": (v_img.name, v_img.getvalue(), v_img.type)},
            timeout=180,
        )
        if resp.ok:
            st.json(resp.json())
        else:
            st.error(resp.text)

with tab_docs:
    up = st.file_uploader("Upload document", type=["pdf", "txt", "jpg", "jpeg", "png"])
    if st.button("Upload Document") and up is not None:
        resp = requests.post(
            f"{BACKEND_URL}/documents/upload",
            files={"file": (up.name, up.getvalue(), up.type or "application/octet-stream")},
            timeout=60,
        )
        if resp.ok:
            st.success("Uploaded successfully.")
            st.json(resp.json())
        else:
            st.error(resp.text)

    if docs:
        for d in docs:
            col1, col2, col3 = st.columns([4, 2, 2])
            col1.write(f"**{d['filename']}** (`{d['document_id']}`)")
            col2.write(f"Status: `{d.get('extraction_status')}` | Chunks: {d.get('chunk_count', 0)}")
            if col3.button("Delete", key=f"del_{d['document_id']}"):
                st.session_state["confirm_delete_id"] = d["document_id"]
                st.session_state["confirm_delete_name"] = d["filename"]

        if st.session_state.get("confirm_delete_id"):
            st.warning(
                f"Confirm deletion of **{st.session_state['confirm_delete_name']}**?"
            )
            if st.button("Confirm Delete"):
                del_resp = requests.delete(
                    f"{BACKEND_URL}/documents/{st.session_state['confirm_delete_id']}",
                    timeout=30,
                )
                st.session_state.pop("confirm_delete_id", None)
                st.session_state.pop("confirm_delete_name", None)
                if del_resp.ok:
                    st.success("Deleted.")
                    st.rerun()
                else:
                    st.error(del_resp.text)

with tab_models:
    try:
        sec = requests.get(f"{BACKEND_URL}/system/security", timeout=5)
        if sec.ok:
            st.subheader("Security Posture")
            st.json(sec.json())
    except requests.RequestException:
        pass
