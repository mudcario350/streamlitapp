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
# (When deployed, Streamlit injects .streamlit/secrets.toml into st.secrets)
gcp_creds = st.secrets["gcp"]
OPENAI_API_KEY = st.secrets["openai"]["api_key"]
if not OPENAI_API_KEY:
    st.error("OpenAI API key not found in st.secrets!"); st.stop()
openai.api_key = OPENAI_API_KEY

# --- Google Sheets Setup ---
def get_worksheet():
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_info(gcp_creds, scopes=scopes)
    client = gspread.authorize(creds)
    return client.open("n8nTest").sheet1

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
    resp = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt_text}]
    )
    return resp.choices[0].message.content.strip()

# --- Append to Google Sheet ---
def append_to_sheet(user_response, evaluation):
    ws = get_worksheet()
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = [
        ts,
        user_response,
        evaluation["creativity"]["score"],
        evaluation["creativity"]["comments"],
        evaluation["insightfulness"]["score"],
        evaluation["insightfulness"]["comments"],
        evaluation["relevance"]["score"],
        evaluation["relevance"]["comments"],
    ]
    ws.append_row(row)

# --- Load stored data ---
def load_data_from_sheet():
    ws = get_worksheet()
    headers = [
        "Timestamp", "User Response",
        "Creativity Score", "Creativity Comments",
        "Insightfulness Score", "Insightfulness Comments",
        "Relevance Score", "Relevance Comments"
    ]
    if ws.row_values(1) != headers:
        ws.update(values=[headers], range_name="A1:H1")
    records = ws.get_all_records(expected_headers=headers)
    return pd.DataFrame(records) if records else pd.DataFrame(columns=headers)

# --- Build evaluation DataFrame ---
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

# --- Streamlit App UI ---
st.title("AI Ethics Evaluation App with Google Sheets Storage")
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
                st.error("Failed to parse GPT output:"); st.code(raw_eval); evaluation = None
            if evaluation:
                append_to_sheet(user_response, evaluation)
                st.success("Your response has been evaluated and stored!")
                st.write("### Evaluation Result")
                df_eval = build_evaluation_df(evaluation)
                for _, row in df_eval.iterrows():
                    st.write(f"**{row['Criterion']}**: Score {row['Score']}")
                    st.text_area(f"{row['Criterion']} Comments", value=row["Comments"],
                                 height=150, disabled=True, key=row["Criterion"])

# --- Display stored responses & averages ---
st.subheader("Stored Responses from Google Sheets")
df = load_data_from_sheet()
if not df.empty:
    for col in ["Creativity Score", "Insightfulness Score", "Relevance Score"]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    st.dataframe(df, use_container_width=True)

    avg = df[["Creativity Score", "Insightfulness Score", "Relevance Score"]]\
          .mean().reset_index()
    avg.columns = ["Criterion", "Average Score"]
    st.subheader("Average Scores Across All Responses")
    chart = alt.Chart(avg).mark_bar().encode(
        x=alt.X("Criterion", sort=None),
        y="Average Score"
    ).properties(width=600)
    st.altair_chart(chart, use_container_width=True)
else:
    st.info("No responses stored in Google Sheets yet.")

