from __future__ import annotations

import json
import os
from urllib import error, request

import streamlit as st


BACKEND_URL = os.getenv("SMARTFIN_BACKEND_URL", "http://localhost:8000")


def _http_json(url: str, *, payload: dict | None = None) -> tuple[bool, dict]:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = request.Request(url, data=data, headers=headers, method="POST" if payload is not None else "GET")
    try:
        with request.urlopen(req, timeout=20) as response:
            return True, json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        return False, {"error": f"HTTP {exc.code}", "detail": detail}
    except Exception as exc:
        return False, {"error": str(exc)}


st.set_page_config(page_title="SmartFin Frontend", layout="wide")
st.title("SmartFin Frontend")
st.caption(f"Backend: {BACKEND_URL}")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Backend Health")
    if st.button("Check backend"):
        ok, response = _http_json(f"{BACKEND_URL}/health")
        if ok:
            st.success("Backend is healthy")
            st.json(response)
        else:
            st.error("Backend health check failed")
            st.json(response)

with col2:
    st.subheader("Analyze request")
    message = st.text_area(
        "Message",
        value="Show me my spending trends and highlight anomalies.",
        height=140,
    )
    monthly_income = st.number_input("Monthly income", min_value=0.0, value=3200.0, step=100.0)
    use_sample_data = st.checkbox("Use sample transactions", value=True)

    if st.button("Run analysis"):
        ok, response = _http_json(
            f"{BACKEND_URL}/analyze",
            payload={
                "message": message,
                "monthly_income": monthly_income,
                "use_sample_data": use_sample_data,
            },
        )
        if ok:
            st.success("Analysis completed")
            st.json(response)
        else:
            st.error("Analysis failed")
            st.json(response)
