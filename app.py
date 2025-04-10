#!/usr/bin/env python3

import os
import json
import datetime
import pandas as pd
import altair as alt
import streamlit as st
import openai
import gspread
from google.oauth2.service_account import Credentials
from dotenv import load_dotenv

# Load environment variables from .env (for local development)
load_dotenv()

# --- Configuration ---
QUESTION = "What are the key ethical considerations when deploying AI systems in society?"

# Get the OpenAI API key from environment variables.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise Exception("OPENAI_API_KEY environment variable not set!")
openai.api_key = OPENAI_API_KEY

# Use the model id as desired.
MODEL_NAME = "gpt-4o-mini"

# --- Google Sheets Setup ---
def get_worksheet():
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]
    credentials_json = os.environ.get("GCP_CREDENTIALS")
    if credentials_json:
        creds_info = json.loads(credentials_json)
        # Fix the private key formatting if needed:
        if "private_key" in creds_info:
            creds_info["private_key"] = creds_info["private_key"].replace("\\n", "\n")
        creds = Credentials.from_service_account_info(creds_info, scopes=scopes)
    else:
        # Fallback for local development: read from file.
        creds_file = "mystic-castle-452716-r3-06468c3f2be8.json"
        creds = Credentials.from_service_account_file(creds_file, scopes=scopes)
    client = gspread.authorize(creds)
    spreadsheet = client.open("n8nTest")
    worksheet = spreadsheet.sheet1
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
    answer_text = response.choices[0].message.content.strip()
    return answer_text

# --- Function to Append Data to Google Sheets ---
def append_to_sheet(user_response, evaluation):
    worksheet = get_worksheet()
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

# --- Function to Load Data from Google Sheets and Ensure Correct Headers ---
def load_data_from_sheet():
    worksheet = get_worksheet()
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
    if records:
        df = pd.DataFrame(records)
    else:
        df = pd.DataFrame(columns=expected_headers)
    return df

# --- Helper Function to Build an Evaluation DataFrame ---
def build_evaluation_df(evaluation):
    data = {
        "Criterion": ["Creativity", "Insightfulness", "Relevance"],
        "Score": [evaluation["creativity"]["score"],
                  evaluation["insightfulness"]["score"],
                  evaluation["relevance"]["score"]],
        "Comments": [evaluation["creativity"]["comments"],
                     evaluation["insightfulness"]["comments"],
                     evaluation["relevance"]["comments"]]
    }
    return pd.DataFrame(data)

# --- Streamlit App UI ---
st.title("AI Ethics Evaluation App with Google Sheets Storage")
st.write(f"**Question:** {QUESTION}")

user_response = st.text_area("Enter your answer to the question above:")

if st.button("Submit Response"):
    if not user_response.strip():
        st.error("Please enter a non-empty response.")
    else:
        with st.spinner("Evaluating your response..."):
            evaluation_json = call_chatgpt(user_response)
            try:
                evaluation = json.loads(evaluation_json)
            except Exception as e:
                st.error("Failed to parse the evaluation response from GPT. Here is the raw output:")
                st.code(evaluation_json)
                evaluation = None
            if evaluation:
                append_to_sheet(user_response, evaluation)
                st.success("Your response has been evaluated and stored in Google Sheets!")
                st.write("### Evaluation Result")
                eval_df = build_evaluation_df(evaluation)
                for idx, row in eval_df.iterrows():
                    st.write(f"**{row['Criterion']}**: Score: {row['Score']}")
                    st.text_area(
                        f"{row['Criterion']} Comments",
                        value=row["Comments"],
                        height=150,
                        disabled=True,
                        key=row["Criterion"]
                    )

# --- Display Stored Responses and Average Scores ---
st.subheader("Stored Responses from Google Sheets")
df = load_data_from_sheet()
if not df.empty:
    for col in ["Creativity Score", "Insightfulness Score", "Relevance Score"]:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype("float64")
    st.dataframe(df, use_container_width=True)
    avg_scores = df[["Creativity Score", "Insightfulness Score", "Relevance Score"]].mean().reset_index()
    avg_scores.columns = ["Criterion", "Average Score"]
    st.subheader("Average Scores Across All Responses")
    chart = alt.Chart(avg_scores).mark_bar().encode(
        x=alt.X("Criterion", sort=None),
        y="Average Score"
    ).properties(width=600)
    st.altair_chart(chart, use_container_width=True)
else:
    st.info("No responses stored in Google Sheets yet.")

