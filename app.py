#!/usr/bin/env python3

import time
import datetime
import streamlit as st
import gspread
import requests
from google.oauth2.service_account import Credentials

# --- Configuration & Secrets ---
if "gcp" not in st.secrets:
    st.error("GCP credentials not found in st.secrets! Please configure GCP secrets.")
    st.stop()
if "n8n" not in st.secrets or "webhook_url" not in st.secrets["n8n"]:
    st.error("n8n webhook URL not found in st.secrets! Please configure under n8n.webhook_url.")
    st.stop()

# Secrets & constants
GCP_CREDS        = st.secrets["gcp"]
SPREADSHEET_NAME = "n8nTest"
N8N_WEBHOOK_URL  = st.secrets["n8n"]["webhook_url"]

# --- Google Sheets Helpers ---

def get_gs_client():
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]
    creds  = Credentials.from_service_account_info(GCP_CREDS, scopes=scopes)
    return gspread.authorize(creds)


def get_worksheet(name, headers):
    client      = get_gs_client()
    spreadsheet = client.open(SPREADSHEET_NAME)
    try:
        ws = spreadsheet.worksheet(name)
    except gspread.exceptions.WorksheetNotFound:
        ws = spreadsheet.add_worksheet(title=name, rows="2000", cols="20")
        ws.update(values=[headers], range_name="A1:{}1".format(chr(ord('A') + len(headers) - 1)))
    return ws


# Worksheets and headers
ANSWERS_SHEET    = "Responses"
EVAL_SHEET       = "Evaluations"
QUESTIONS_SHEET  = "QUESTIONS"
ANSW_HEADERS     = ["RecordID","Timestamp","StudentID","ConversationID","Round","Q1","Q2","Q3"]
EVAL_HEADERS     = ["RecordID","ConversationID","StudentID","Round",
                    "Q1Grade","Q1Feedback","Q2Grade","Q2Feedback",
                    "Q3Grade","Q3Feedback","Average","GeneralFeedback"]


def generate_record_id(ws):
    values = ws.get_all_values()
    if len(values) <= 1:
        return 1
    try:
        ids = [int(row[0]) for row in values[1:] if row[0].isdigit()]
        return max(ids) + 1 if ids else 1
    except Exception:
        return 1


def append_response(student_id, conv_id, round_num, a1, a2, a3):
    ws        = get_worksheet(ANSWERS_SHEET, ANSW_HEADERS)
    record_id = generate_record_id(ws)
    ts        = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ws.append_row([record_id, ts, student_id, conv_id, round_num, a1, a2, a3])
    return record_id


def append_evaluation(record):
    ws = get_worksheet(EVAL_SHEET, EVAL_HEADERS)
    ws.append_row([record.get(h, "") for h in EVAL_HEADERS])


def notify_n8n(payload):
    try:
        requests.post(N8N_WEBHOOK_URL, json=payload, timeout=5)
    except Exception as e:
        st.warning(f"Failed to notify n8n: {e}")


def load_questions():
    ws = get_worksheet(QUESTIONS_SHEET, [])
    recs = ws.get_all_records()
    if not recs:
        st.error("No questions found in QUESTIONS sheet.")
        st.stop()
    last = recs[-1]
    return last.get("Question1", ""), last.get("Question2", ""), last.get("Question3", "")


def poll_for_evaluation(conv_id, round_num, since_row, interval=1, timeout=300):
    ws    = get_worksheet(EVAL_SHEET, EVAL_HEADERS)
    start = time.time()
    while True:
        if time.time() - start > timeout:
            return None
        vals   = ws.get_all_values()
        header = vals[0]
        for idx, row in enumerate(vals[1:], start=2):
            if row[1] == str(conv_id) and int(row[3]) == round_num and idx > since_row:
                return dict(zip(header, row))
        time.sleep(interval)

# --- Streamlit UI ---
st.set_page_config(page_title="AI Ethics Peer-Review", layout="centered")
st.title("AI Ethics Peer-Review")

# Student ID
student_id = st.text_input("Enter your Student ID:").strip()
if not student_id:
    st.info("Please enter your Student ID to continue.")
    st.stop()

# Load questions
q1, q2, q3 = load_questions()

# Initialize session state
if "conv_id" not in st.session_state:
    st.session_state.conv_id  = f"C-{student_id}-{int(time.time())}"
if "round" not in st.session_state:
    st.session_state.round    = 1
if "last_eval_row" not in st.session_state:
    st.session_state.last_eval_row = 1

st.write(f"### Round {st.session_state.round}")
st.write(f"**Question 1:** {q1}")
answer1 = st.text_area("Response Q1", key=f"a1_{st.session_state.round}")
st.write(f"**Question 2:** {q2}")
answer2 = st.text_area("Response Q2", key=f"a2_{st.session_state.round}")
st.write(f"**Question 3:** {q3}")
answer3 = st.text_area("Response Q3", key=f"a3_{st.session_state.round}")

if st.button("Submit"):    
    if not answer1.strip():
        st.error("Q1 cannot be empty.")
    else:
        # record response
        record_id = append_response(student_id, st.session_state.conv_id,
                                    st.session_state.round,
                                    answer1, answer2, answer3)
        # send for evaluation
        payload = {"record_id": record_id,
                   "conversation_id": st.session_state.conv_id,
                   "student_id": student_id,
                   "round": st.session_state.round,
                   "answers": {"q1": answer1, "q2": answer2, "q3": answer3}}
        notify_n8n(payload)
        # poll
        with st.spinner("Waiting for evaluation…"):
            eval_row = poll_for_evaluation(st.session_state.conv_id,
                                           st.session_state.round,
                                           st.session_state.last_eval_row)
        if not eval_row:
            st.error("Evaluation timed out. Try refreshing.")
        else:
            st.success("Evaluation received!")
            # display
            for i in [1,2,3]:
                st.write(f"**Q{i} Grade:** {eval_row.get(f'Q{i}Grade','')}")
                st.text_area(f"Feedback Q{i}",
                              value=eval_row.get(f'Q{i}Feedback',''),
                              height=120, disabled=True)
            st.write(f"**Average:** {eval_row.get('Average','')}")
            st.text_area("General Feedback",
                          value=eval_row.get('GeneralFeedback',''),
                          height=150, disabled=True)
            # save eval record row index
            st.session_state.last_eval_row += 1
            # prepare for next round
            st.session_state.round += 1
            st.experimental_rerun()

# Manual refresh fallback
st.markdown("---")
st.header("Manual Refresh Evaluation")
if st.button("Refresh Evaluation"):
    refreshed = poll_for_evaluation(st.session_state.conv_id,
                                     st.session_state.round-1,
                                     0, timeout=1)
    if not refreshed:
        st.warning("No new evaluation yet.")
    else:
        st.success("Loaded latest evaluation")
        for i in [1,2,3]:
            st.write(f"**Q{i} Grade:** {refreshed.get(f'Q{i}Grade','')}")
            st.text_area(f"Feedback Q{i}",
                          value=refreshed.get(f'Q{i}Feedback',''),
                          height=120, disabled=True)
        st.write(f"**Average:** {refreshed.get('Average','')}")
        st.text_area("General Feedback",
                      value=refreshed.get('GeneralFeedback',''),
                      height=150, disabled=True)
