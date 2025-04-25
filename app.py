#!/usr/bin/env python3

import json
import datetime
import pandas as pd
import altair as alt
import streamlit as st
import openai
import gspread
from google.oauth2.service_account import Credentials

# --- Configuration / Secrets ---
# GCP credentials under [gcp] and OpenAI API key under [openai] in .streamlit/secrets.toml
if "gcp" not in st.secrets or "openai" not in st.secrets:
    st.error("Required secrets not found in st.secrets! Please configure GCP and OpenAI secrets.")
    st.stop()

gcp_creds = st.secrets["gcp"]
OPENAI_API_KEY = st.secrets["openai"].get("api_key", "")
if not OPENAI_API_KEY:
    st.error("OpenAI API key not found in st.secrets!")
    st.stop()

openai.api_key = OPENAI_API_KEY
MODEL_NAME    = st.secrets["openai"].get("model_name", "gpt-4o-mini")
SPREADSHEET_NAME = "n8nTest"

# --- Helper Functions ---
def get_gs_client():
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]
    creds = Credentials.from_service_account_info(gcp_creds, scopes=scopes)
    return gspread.authorize(creds)

def get_student_worksheet(student_id):
    client      = get_gs_client()
    spreadsheet = client.open(SPREADSHEET_NAME)
    try:
        return spreadsheet.worksheet(student_id)
    except gspread.exceptions.WorksheetNotFound:
        return spreadsheet.add_worksheet(title=student_id, rows="100", cols="20")

def load_questions():
    client      = get_gs_client()
    spreadsheet = client.open(SPREADSHEET_NAME)
    try:
        q_sheet = spreadsheet.worksheet("QUESTIONS")
    except gspread.exceptions.WorksheetNotFound:
        st.error("QUESTIONS sheet not found in the spreadsheet.")
        st.stop()

    records = q_sheet.get_all_records()
    if not records:
        st.error("No questions found in QUESTIONS sheet.")
        st.stop()

    last = records[-1]
    return (
        last.get("Question1", ""),
        last.get("Question2", ""),
        last.get("Question3", "")
    )

def call_chatgpt(user_response, question):
    prompt_text = f"""<PromptForGPT>
<response>
{user_response}
</response>
<question>
{question}
</question>
<prompt>
Please evaluate how effectively the response answered the question. Judge the response on three criteria, creativity, insightfulness, and relevance. Come up with comments that could be used to improve the response for each of these criteria, and also come up with a numerical grade out of 10, you can use up to one decimal place for each. Try to be thoughtful in your evaluations, and do not be afraid to give the response low numerical grades if it is not creative, insightful, or relevant.
</prompt>
<output_format>
{{"creativity": {{"score": [score for creativity], "comments": "[comments for creativity]"}}, "insightfulness": {{"score": [score for insightfulness], "comments": "[comments for insightfulness]"}}, "relevance": {{"score": [score for relevance], "comments": "[comments for relevance]"}}}}
</output_format>
</PromptForGPT>"""

    resp = openai.ChatCompletion.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt_text}]
    )
    return resp.choices[0].message.content.strip()

def append_to_sheet(student_id, user_response, evaluation):
    ws        = get_student_worksheet(student_id)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_row   = [
        timestamp,
        user_response,
        evaluation["creativity"]["score"],
        evaluation["creativity"]["comments"],
        evaluation["insightfulness"]["score"],
        evaluation["insightfulness"]["comments"],
        evaluation["relevance"]["score"],
        evaluation["relevance"]["comments"]
    ]
    ws.append_row(new_row)

def load_data_from_sheet(student_id):
    ws = get_student_worksheet(student_id)
    expected_headers = [
        "Timestamp",
        "User Response",
        "Creativity Score",
        "Creativity Comments",
        "Insightfulness Score",
        "Insightfulness Comments",
        "Relevance Score",
        "Relevance Comments"
    ]
    current = ws.row_values(1)
    if current != expected_headers:
        ws.update(values=[expected_headers], range_name="A1:H1")
    records = ws.get_all_records(expected_headers=expected_headers)
    return pd.DataFrame(records) if records else pd.DataFrame(columns=expected_headers)

def build_evaluation_df(evaluation):
    return pd.DataFrame({
        "Criterion": ["Creativity", "Insightfulness", "Relevance"],
        "Score": [
            evaluation["creativity"]["score"],
            evaluation["insightfulness"]["score"],
            evaluation["relevance"]["score"]
        ],
        "Comments": [
            evaluation["creativity"]["comments"],
            evaluation["insightfulness"]["comments"],
            evaluation["relevance"]["comments"]
        ]
    })

# --- Load dynamic questions from the QUESTIONS sheet ---
q1, q2, q3 = load_questions()

# --- Streamlit App UI ---
student_id = st.text_input("Please enter your student ID:").strip()
if not student_id:
    st.info("Enter your student ID to proceed.")
    st.stop()

st.write(f"### Welcome, student {student_id}!")

# Show all three questions
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
        with st.spinner("Evaluating your response to Question 1..."):
            raw_eval = call_chatgpt(answer1, q1)
            try:
                evaluation = json.loads(raw_eval)
            except json.JSONDecodeError:
                st.error("Failed to parse the evaluation response from GPT.")
                st.code(raw_eval)
                evaluation = None

            if evaluation:
                append_to_sheet(student_id, answer1, evaluation)
                st.success("Your response has been evaluated and stored!")
                st.write("### Evaluation Result")
                df_eval = build_evaluation_df(evaluation)
                for _, row in df_eval.iterrows():
                    st.write(f"**{row['Criterion']}**: Score: {row['Score']}")
                    st.text_area(
                        f"{row['Criterion']} Comments",
                        value=row["Comments"],
                        height=150,
                        disabled=True,
                        key=row["Criterion"]
                    )

# --- Display stored responses & averages ---
st.subheader(f"Stored Responses for Student ID: {student_id}")
df = load_data_from_sheet(student_id)
if not df.empty:
    for col in ["Creativity Score", "Insightfulness Score", "Relevance Score"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    st.dataframe(df, use_container_width=True)

    avg_scores = (
        df[["Creativity Score", "Insightfulness Score", "Relevance Score"]]
        .mean()
        .reset_index()
    )
    avg_scores.columns = ["Criterion", "Average Score"]

    st.subheader("Average Scores Across Your Responses")
    chart = (
        alt.Chart(avg_scores)
        .mark_bar()
        .encode(x=alt.X("Criterion", sort=None), y="Average Score")
        .properties(width=600)
    )
    st.altair_chart(chart, use_container_width=True)
else:
    st.info("No responses stored for this student ID yet.")

