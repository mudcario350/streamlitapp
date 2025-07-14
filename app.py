#!/usr/bin/env python3

import time
import datetime
import pandas as pd
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

# service account info
gcp_creds        = st.secrets["gcp"]
SPREADSHEET_NAME = "n8nTest"
# n8n webhook endpoint
N8N_WEBHOOK_URL  = st.secrets["n8n"]["webhook_url"]

# --- Google Sheets Helpers ---
def get_gs_client():
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]
    creds  = Credentials.from_service_account_info(gcp_creds, scopes=scopes)
    return gspread.authorize(creds)

def get_r1_answers_worksheet():
    client      = get_gs_client()
    spreadsheet = client.open(SPREADSHEET_NAME)
    try:
        ws = spreadsheet.worksheet("R1 Answers")
    except gspread.exceptions.WorksheetNotFound:
        ws = spreadsheet.add_worksheet(title="R1 Answers", rows="1000", cols="10")
        headers = [
            "Record ID",
            "Timestamp",
            "Student ID",
            "Question1_Answer",
            "Question2_Answer",
            "Question3_Answer"
        ]
        ws.update(values=[headers], range_name="A1:F1")
    return ws

def get_r1_evaluation_worksheet():
    client      = get_gs_client()
    spreadsheet = client.open(SPREADSHEET_NAME)
    try:
        ws = spreadsheet.worksheet("R1 Evaluation")
    except gspread.exceptions.WorksheetNotFound:
        ws = spreadsheet.add_worksheet(title="R1 Evaluation", rows="1000", cols="12")
        headers = [
            "Record ID",
            "Student ID",
            "Answer1 Grade", "Answer1 Feedback",
            "Answer2 Grade", "Answer2 Feedback",
            "Answer3 Grade", "Answer3 Feedback",
            "Average", "General Feedback"
        ]
        ws.update(values=[headers], range_name="A1:J1")
    return ws

def generate_record_id(ws):
    """
    Generates a unique integer ID for a new record by finding the max existing ID and adding 1.
    """
    values = ws.get_all_values()  # includes header
    if len(values) <= 1:
        return 1
    try:
        existing_ids = [int(row[0]) for row in values[1:] if row[0].isdigit()]
        return max(existing_ids) + 1 if existing_ids else 1
    except Exception:
        return 1

def append_to_r1_answers(student_id, a1, a2, a3):
    ws = get_r1_answers_worksheet()
    record_id = generate_record_id(ws)
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ws.append_row([record_id, ts, student_id, a1, a2, a3])
    return record_id

def notify_n8n(record_id, student_id, a1, a2, a3):
    """Send a POST to your n8n webhook to trigger downstream workflows."""
    payload = {
        "record_id": record_id,
        "student_id": student_id,
        "answers": {"q1": a1, "q2": a2, "q3": a3}
    }
    try:
        requests.post(N8N_WEBHOOK_URL, json=payload, timeout=5)
    except Exception as e:
        st.warning(f"Failed to notify n8n: {e}")

def get_last_evaluation_row_index(record_id):
    ws     = get_r1_evaluation_worksheet()
    values = ws.get_all_values()  # includes header
    last_i = 1  # header row
    for idx, row in enumerate(values[1:], start=2):
        if str(row[0]) == str(record_id):
            last_i = max(last_i, idx)
    return last_i

def poll_for_evaluation(record_id, since_row, interval=1, timeout=300):
    """
    Polls every `interval` seconds until a new row for `record_id` appears
    beyond `since_row`. Times out after `timeout` seconds.
    """
    ws    = get_r1_evaluation_worksheet()
    start = time.time()
    while True:
        if time.time() - start > timeout:
            return None
        vals   = ws.get_all_values()
        header = vals[0]
        for idx, row in enumerate(vals[1:], start=2):
            if str(row[0]) == str(record_id) and idx > since_row:
                return dict(zip(header, row))
        time.sleep(interval)

# --- Streamlit UI ---
st.title("AI Ethics Peer-Review")

student_id = st.text_input("Please enter your student ID:").strip()
if not student_id:
    st.info("Enter your student ID to proceed.")
    st.stop()

# load dynamic questions
q1, q2, q3 = load_questions()

st.write(f"### Welcome, student {student_id}!")
st.write(f"**Question 1:** {q1}")
answer1 = st.text_area("Your response to Question 1:")

st.write(f"**Question 2:** {q2}")
answer2 = st.text_area("Your response to Question 2:")

st.write(f"**Question 3:** {q3}")
answer3 = st.text_area("Your response to Question 3:")

if st.button("Submit Responses"):
    if not answer1.strip():
        st.error("Please enter a non-empty response for Question 1.")
    else:
        # append your answers and get record_id
        record_id = append_to_r1_answers(student_id, answer1, answer2, answer3)
        # notify n8n via webhook
        notify_n8n(record_id, student_id, answer1, answer2, answer3)

        # record where we were in the evaluation sheet for this record
        last_idx = get_last_evaluation_row_index(record_id)

        # now poll for the new evaluation row
        with st.spinner("Waiting for your evaluation from n8n…"):
            eval_row = poll_for_evaluation(record_id, last_idx)
        if not eval_row:
            st.error("Timed out waiting for evaluation. You can still use 'Refresh Evaluation' below.")
        else:
            st.success("Evaluation received!")
            # render it immediately
            for i in [1,2,3]:
                grade = eval_row.get(f"Answer{i} Grade", "")
                fb    = eval_row.get(f"Answer{i} Feedback", "")
                st.write(f"**Question {i} Grade:** {grade}")
                st.text_area(
                    f"Feedback for Question {i}",
                    value=fb,
                    height=150,
                    disabled=True,
                    key=f"Q{i}"
                )
            avg = eval_row.get("Average", "")
            gf  = eval_row.get("General Feedback", "")
            st.write(f"**Average Score:** {avg}")
            st.text_area(
                "General Feedback",
                value=gf,
                height=150,
                disabled=True,
                key="general"
            )

st.markdown("---")
st.header("Refresh Evaluation (manual fallback)")

if st.button("Refresh Evaluation"):
    row = poll_for_evaluation(record_id, 0, interval=1, timeout=1)
    if not row:
        st.warning("No evaluation found for your record yet.")
    else:
        st.success("Loaded your latest evaluation!")
        for i in [1,2,3]:
            grade = row.get(f"Answer{i} Grade", "")
            fb    = row.get(f"Answer{i} Feedback", "")
            st.write(f"**Question {i} Grade:** {grade}")
            st.text_area(
                f"Feedback for Question {i}",
                value=fb,
                height=150,
                disabled=True,
                key=f"R{i}"
            )
        st.write(f"**Average Score:** {row.get('Average','')}")
        st.text_area(
            "General Feedback",
            value=row.get("General Feedback",""),
            height=150,
            disabled=True,
            key="Rgeneral"
        )



'''
import time
import datetime
import pandas as pd
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials

# --- Configuration & Secrets ---
if "gcp" not in st.secrets:
    st.error("GCP credentials not found in st.secrets! Please configure GCP secrets.")
    st.stop()

gcp_creds        = st.secrets["gcp"]
SPREADSHEET_NAME = "n8nTest"

# --- Google Sheets Helpers ---
def get_gs_client():
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]
    creds  = Credentials.from_service_account_info(gcp_creds, scopes=scopes)
    return gspread.authorize(creds)

def get_r1_answers_worksheet():
    client      = get_gs_client()
    spreadsheet = client.open(SPREADSHEET_NAME)
    try:
        ws = spreadsheet.worksheet("R1 Answers")
    except gspread.exceptions.WorksheetNotFound:
        ws = spreadsheet.add_worksheet(title="R1 Answers", rows="1000", cols="10")
        headers = ["Timestamp", "ID", "Question1_Answer", "Question2_Answer", "Question3_Answer"]
        ws.update(values=[headers], range_name="A1:E1")
    return ws

def get_r1_evaluation_worksheet():
    client      = get_gs_client()
    spreadsheet = client.open(SPREADSHEET_NAME)
    try:
        ws = spreadsheet.worksheet("R1 Evaluation")
    except gspread.exceptions.WorksheetNotFound:
        ws = spreadsheet.add_worksheet(title="R1 Evaluation", rows="1000", cols="10")
        headers = [
            "ID",
            "Answer1 Grade", "Answer1 Feedback",
            "Answer2 Grade", "Answer2 Feedback",
            "Answer3 Grade", "Answer3 Feedback",
            "Average", "General Feedback"
        ]
        ws.update(values=[headers], range_name="A1:I1")
    return ws

def load_questions():
    client      = get_gs_client()
    spreadsheet = client.open(SPREADSHEET_NAME)
    try:
        q_sheet = spreadsheet.worksheet("QUESTIONS")
    except gspread.exceptions.WorksheetNotFound:
        st.error("QUESTIONS sheet not found.")
        st.stop()
    recs = q_sheet.get_all_records()
    if not recs:
        st.error("No questions found in QUESTIONS sheet.")
        st.stop()
    last = recs[-1]
    return (
        last.get("Question1", ""),
        last.get("Question2", ""),
        last.get("Question3", "")
    )

def append_to_r1_answers(student_id, a1, a2, a3):
    ws = get_r1_answers_worksheet()
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ws.append_row([ts, student_id, a1, a2, a3])

def get_last_evaluation_row_index(student_id):
    ws     = get_r1_evaluation_worksheet()
    values = ws.get_all_values()  # includes header
    last_i = 1  # header row
    for idx, row in enumerate(values[1:], start=2):
        if str(row[0]) == str(student_id):
            last_i = max(last_i, idx)
    return last_i

def poll_for_evaluation(student_id, since_row, interval=1, timeout=300):
    """
    Polls every `interval` seconds (now 1s) until a new row for `student_id` appears
    beyond `since_row`. Times out after `timeout` seconds.
    """
    ws    = get_r1_evaluation_worksheet()
    start = time.time()
    while True:
        if time.time() - start > timeout:
            return None
        vals   = ws.get_all_values()
        header = vals[0]
        for idx, row in enumerate(vals[1:], start=2):
            if str(row[0]) == str(student_id) and idx > since_row:
                return dict(zip(header, row))
        time.sleep(interval)

# --- Streamlit UI ---
st.title("AI Ethics Peer-Review")

student_id = st.text_input("Please enter your student ID:").strip()
if not student_id:
    st.info("Enter your student ID to proceed.")
    st.stop()

# load dynamic questions
q1, q2, q3 = load_questions()

st.write(f"### Welcome, student {student_id}!")
st.write(f"**Question 1:** {q1}")
answer1 = st.text_area("Your response to Question 1:")

st.write(f"**Question 2:** {q2}")
answer2 = st.text_area("Your response to Question 2:")

st.write(f"**Question 3:** {q3}")
answer3 = st.text_area("Your response to Question 3:")

if st.button("Submit Responses"):
    if not answer1.strip():
        st.error("Please enter a non-empty response for Question 1.")
    else:
        # record where we were in the evaluation sheet
        last_idx = get_last_evaluation_row_index(student_id)
        # append your answers
        append_to_r1_answers(student_id, answer1, answer2, answer3)

        # now poll for the new evaluation row
        with st.spinner("Waiting for your evaluation from n8n…"):
            eval_row = poll_for_evaluation(student_id, last_idx)  # now checks every 1s
        if not eval_row:
            st.error("Timed out waiting for evaluation. You can still use 'Refresh Evaluation' below.")
        else:
            st.success("Evaluation received!")
            # render it immediately
            for i in [1,2,3]:
                grade = eval_row.get(f"Answer{i} Grade", "")
                fb    = eval_row.get(f"Answer{i} Feedback", "")
                st.write(f"**Question {i} Grade:** {grade}")
                st.text_area(
                    f"Feedback for Question {i}",
                    value=fb,
                    height=150,
                    disabled=True,
                    key=f"Q{i}"
                )
            avg = eval_row.get("Average", "")
            gf  = eval_row.get("General Feedback", "")
            st.write(f"**Average Score:** {avg}")
            st.text_area(
                "General Feedback",
                value=gf,
                height=150,
                disabled=True,
                key="general"
            )

st.markdown("---")
st.header("Refresh Evaluation (manual fallback)")

if st.button("Refresh Evaluation"):
    row = poll_for_evaluation(student_id, 0, interval=1, timeout=1)
    if not row:
        st.warning("No evaluation found for your ID yet.")
    else:
        st.success("Loaded your latest evaluation!")
        for i in [1,2,3]:
            grade = row.get(f"Answer{i} Grade", "")
            fb    = row.get(f"Answer{i} Feedback", "")
            st.write(f"**Question {i} Grade:** {grade}")
            st.text_area(
                f"Feedback for Question {i}",
                value=fb,
                height=150,
                disabled=True,
                key=f"R{i}"
            )
        st.write(f"**Average Score:** {row.get('Average','')}")
        st.text_area(
            "General Feedback",
            value=row.get("General Feedback",""),
            height=150,
            disabled=True,
            key="Rgeneral"
        )

# --- Previously: bottom-of-page table & chart is commented out ---
# st.subheader(f"Stored Evaluations for Student ID: {student_id}")
# …

'''
