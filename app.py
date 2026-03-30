import streamlit as st
from src.predict import predict_sequence

st.title("🔐 Real-Time Login IDS")

st.write("Simulate login attempts and detect brute-force attacks")

# -------- SESSION STATE --------
if "fail_count" not in st.session_state:
    st.session_state.fail_count = 0

# -------- USER NORMAL PROFILE --------
NORMAL_PROFILE = ["E1", "E2", "E3"]

# -------- LOGIN FORM --------
username = st.text_input("Username")
password = st.text_input("Password", type="password")

# -------- LOGIN BUTTON --------
if st.button("Login"):

    if username == "admin" and password == "1234":
        st.success("✅ Login Successful")
        st.session_state.fail_count = 0

        log_seq = NORMAL_PROFILE

    else:
        st.error("❌ Wrong Credentials")
        st.session_state.fail_count += 1

        # Progressive log behavior
        log_seq = NORMAL_PROFILE + ["E91"] * st.session_state.fail_count

    # -------- IDS DETECTION (MODEL) --------
    pred, score = predict_sequence(log_seq)

    st.write(f"Log sequence: {log_seq}")
    st.write(f"Score: {score:.3f}")

    # -------- HYBRID DECISION (MODEL + BEHAVIOR) --------
    if pred == 0:
        st.success("✅ Normal Behavior")

    elif pred == 1 and st.session_state.fail_count <= 2:
        st.warning("🟡 Minor Anomaly Detected")

    elif pred == 1 and st.session_state.fail_count <= 4:
        st.warning("🟠 Suspicious Activity")

    else:
        st.error("🚨 ATTACK DETECTED (Brute Force)")

    # -------- RISK METER (BONUS) --------
    risk = min(score * 100, 100)
    st.progress(int(risk))
    st.write(f"Risk Level: {risk:.1f}%")

# -------- EXTRA INFO --------
st.write(f"Failed Attempts: {st.session_state.fail_count}")

# -------- RESET BUTTON --------
if st.button("Reset System"):
    st.session_state.fail_count = 0
    st.success("System Reset")