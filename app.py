#!/usr/bin/env python3

import json
import datetime
import pandas as pd
import altair as alt
import streamlit as st
import openai
import gspread
from google.oauth2.service_account import Credentials

# --- Configuration ---
QUESTION = "What are the key ethical considerations when deploying AI systems in society?"

# --- Load secrets ---
# GCP credentials under [gcp] and OpenAI API key under [openai] in .streamlit/secrets.toml
if "gcp" not in st.secrets or "openai" not in st.secrets:
    st.error("Required secrets not found in st.secrets! Please configure GCP and OpenAI secrets.")
    st.stop()

gcp_creds = st.secrets["gcp"]
OPENAI_API_KEY = st.secrets["openai"]["api_key"]
if not OPENAI_API_KEY:
    st.error("OpenAI API key not found in st.secrets!")
    st.stop()
openai.api_key = OPENAI_API_KEY
MODEL_NAME = st.secrets.get("openai", {}).get("model_name", "gpt-4o-mini")

# --- Google Sheets Setup: Worksheet per Student ---
def get_student_worksheet(student_id):
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]
    creds = Credentials.from_service_account_info(gcp_creds, scopes=scopes)
    client = gspread.authorize(creds)
    spreadsheet = client.open("n8nTest")
    try:
        worksheet = spreadsheet.worksheet(student_id)
    except gspread.exceptions.WorksheetNotFound:
        worksheet = spreadsheet.add_worksheet(title=student_id, rows="100", cols="20")
    return worksheet

# --- Function to Call OpenAI Chat API ---
def call_chatgpt(user_response):
    prompt_text = f"""<PromptForGPT>
<response>
{user_response}
</response>
<question>
{QUESTION}
</question>
<prompt>
Please evaluate how effectively the response answered the question. Judge the response on three criteria, creativity, insightfulness, and relevance. Come up with comments that could be used to improve the response for each of these criteria, and also come up with a numerical grade out of 10, you can use up to one decimal place for each. Try to be thoughtful in your evaluations, and do not be afraid to give the response low numerical grades if it is not creative, insightful, or relevant.
</prompt>
<output_format>
{{"creativity": {{"score": [score for creativity], "comments": "[comments for creativity]"}}, "insightfulness": {{"score": [score for insightfulness], "comments": "[comments for insightfulness]"}}, "relevance": {{"score": [score for relevance], "comments": "[comments for relevance]"}}}}
</output_format>
</PromptForGPT>"""

    response = openai.ChatCompletion.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt_text}]
    )
    return response.choices[0].message.content.strip()

# --- Function to Append Data to the Student's Sheet ---
def append_to_sheet(student_id, user_response, evaluation):
    worksheet = get_student_worksheet(student_id)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_row = [
        timestamp,
        user_response,
        evaluation["creativity"]["score"],
        evaluation["creativity"]["comments"],
        evaluation["insightfulness"]["score"],
        evaluation["insightfulness"]["comments"],
        evaluation["relevance"]["score"],
        evaluation["relevance"]["comments"]
    ]
    worksheet.append_row(new_row)

# --- Function to Load Data from the Student's Sheet and Ensure Correct Headers ---
def load_data_from_sheet(student_id):
    worksheet = get_student_worksheet(student_id)
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
    current_headers = worksheet.row_values(1)
    if current_headers != expected_headers:
        worksheet.update(values=[expected_headers], range_name="A1:H1")
    records = worksheet.get_all_records(expected_headers=expected_headers)
    return pd.DataFrame(records) if records else pd.DataFrame(columns=expected_headers)

# --- Helper Function to Build an Evaluation DataFrame ---
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

# --- Begin App UI ---
# Ask for the student id
student_id = st.text_input("Please enter your student ID:").strip()
if not student_id:
    st.info("Enter your student ID to proceed.")
    st.stop()

# Greet and show question
st.write(f"### Welcome, student {student_id}!")
st.write(f"**Question:** {QUESTION}")
user_response = st.text_area("Enter your answer to the question above:")

if st.button("Submit Response"):
    if not user_response.strip():
        st.error("Please enter a non-empty response.")
    else:
        with st.spinner("Evaluating your response..."):
            raw_eval = call_chatgpt(user_response)
            try:
                evaluation = json.loads(raw_eval)
            except json.JSONDecodeError:
                st.error("Failed to parse the evaluation response from GPT.")
                st.code(raw_eval)
                evaluation = None
            if evaluation:
                append_to_sheet(student_id, user_response, evaluation)
                st.success("Your response has been evaluated and stored in your sheet!")
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

# --- Display Stored Responses and Average Scores for the Student ---
st.subheader(f"Stored Responses for Student ID: {student_id}")
df = load_data_from_sheet(student_id)
if not df.empty:
    for col in ["Creativity Score", "Insightfulness Score", "Relevance Score"]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    st.dataframe(df, use_container_width=True)
    avg_scores = df[["Creativity Score", "Insightfulness Score", "Relevance Score"]] \
        .mean().reset_index()
    avg_scores.columns = ["Criterion", "Average Score"]
    st.subheader("Average Scores Across Your Responses")
    chart = alt.Chart(avg_scores).mark_bar().encode(
        x=alt.X("Criterion", sort=None),
        y="Average Score"
    ).properties(width=600)
    st.altair_chart(chart, use_container_width=True)
else:
    st.info("No responses stored for this student ID yet.")

