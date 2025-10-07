#!/usr/bin/env python3

import time
import datetime
import json
from typing import Dict, Any, Optional, List
import uuid
import threading
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor
import logging

import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState

# Import scroll component
try:
    from streamlit_scroll_to_top import scroll_to_here
    SCROLL_AVAILABLE = True
except ImportError:
    SCROLL_AVAILABLE = False
    st.warning("streamlit-scroll-to-top not installed. Run: pip install streamlit-scroll-to-top")

# ===========================
# Configuration (from config.py)
# ===========================

# App Configuration
SPREADSHEET_NAME = "n8nTest"
THRESHOLD_SCORE = 8.0  # completion threshold (1-10 scale)

# Google Doc IDs for prompts
# Replace these with your actual Google Doc IDs
PROMPT_DOC_IDS = {
    "grading": "YOUR_GRADING_PROMPT_DOC_ID",
    "evaluation": "YOUR_EVALUATION_PROMPT_DOC_ID",
    "conversation": "YOUR_CONVERSATION_PROMPT_DOC_ID"
}

# Prompt names within each document
PROMPT_NAMES = {
    "grading": "grading_prompt",
    "evaluation": "evaluation_prompt", 
    "conversation": "conversation_prompt"
}

def get_openai_api_key():
    """Get OpenAI API key from Streamlit secrets."""
    if "openai" not in st.secrets or "api_key" not in st.secrets["openai"]:
        st.error("OpenAI API key missing. Add under [openai] in secrets.")
        st.stop()
    return st.secrets["openai"]["api_key"]

def get_gcp_credentials():
    """Get GCP credentials from Streamlit secrets."""
    if "gcp" not in st.secrets:
        st.error("GCP credentials missing. Add under [gcp] in secrets.")
        st.stop()
    return st.secrets["gcp"]

def get_gemini_api_key():
    """Get Gemini API key from Streamlit secrets."""
    if "gemini" not in st.secrets or "api_key" not in st.secrets["gemini"]:
        st.error("Gemini API key missing. Add under [gemini] in secrets.")
        st.stop()
    return st.secrets["gemini"]["api_key"]

# LLM Configuration
LLM_PROVIDER = "gemini"  # Options: "openai" or "gemini"
DEFAULT_MODEL = {
    "openai": "gpt-4o-mini-2024-07-18",
    "gemini": "gemini-2.5-flash"
}


# ===========================
# Prompt Manager (from da_prompt_manager.py)
# ===========================

class PromptManager:
    def __init__(self, credentials_dict: dict, sheets_manager=None):
        """Initialize the prompt manager with Google credentials."""
        self.credentials = Credentials.from_service_account_info(
            credentials_dict,
            scopes=['https://www.googleapis.com/auth/documents.readonly']
        )
        self.service = build('docs', 'v1', credentials=self.credentials)
        self.sheets_manager = sheets_manager
        self._prompts_cache = {}
    
    def get_prompt_from_doc(self, doc_id: str, prompt_name: str) -> Optional[str]:
        """
        Extract a specific prompt from a Google Doc.
        
        Args:
            doc_id: Google Doc ID (from URL)
            prompt_name: Name of the prompt to extract (e.g., 'grading_prompt')
        
        Returns:
            The prompt text or None if not found
        """
        try:
            # Get the document content
            document = self.service.documents().get(documentId=doc_id).execute()
            
            # Extract text content
            content = document.get('body', {}).get('content', [])
            full_text = self._extract_text_from_content(content)
            
            # Parse prompts (assuming format like "GRADING_PROMPT: ...")
            prompts = self._parse_prompts_from_text(full_text)
            
            return prompts.get(prompt_name)
            
        except Exception as e:
            st.error(f"Error loading prompt '{prompt_name}' from Google Doc: {e}")
            return None
    
    def _extract_text_from_content(self, content: list) -> str:
        """Extract text from Google Doc content structure."""
        text = ""
        for element in content:
            if 'paragraph' in element:
                for para_element in element['paragraph']['elements']:
                    if 'textRun' in para_element:
                        text += para_element['textRun']['content']
        return text
    
    def _parse_prompts_from_text(self, text: str) -> Dict[str, str]:
        """Parse prompts from text using a simple format."""
        prompts = {}
        current_prompt = None
        current_content = []
        
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if this line starts a new prompt (format: PROMPT_NAME:)
            if ':' in line and line.split(':')[0].isupper():
                # Save previous prompt if exists
                if current_prompt:
                    prompts[current_prompt] = '\n'.join(current_content).strip()
                
                # Start new prompt
                current_prompt = line.split(':')[0].lower()
                current_content = []
                
                # Add the rest of the line as content if it exists
                remaining = line.split(':', 1)[1].strip()
                if remaining:
                    current_content.append(remaining)
            else:
                # Add line to current prompt content
                if current_prompt:
                    current_content.append(line)
        
        # Save the last prompt
        if current_prompt:
            prompts[current_prompt] = '\n'.join(current_content).strip()
        
        return prompts
    
    def get_cached_prompt(self, doc_id: str, prompt_name: str) -> Optional[str]:
        """Get a prompt with caching to avoid repeated API calls."""
        cache_key = f"{doc_id}_{prompt_name}"
        
        if cache_key not in self._prompts_cache:
            self._prompts_cache[cache_key] = self.get_prompt_from_doc(doc_id, prompt_name)
        
        return self._prompts_cache[cache_key]
    
    def get_prompt_from_assignment(self, assignment_id: str, prompt_type: str) -> Optional[str]:
        """
        Get a prompt from the Google Sheets assignments table.
        
        Args:
            assignment_id: The assignment ID to look up
            prompt_type: Type of prompt ("grading" or "conversation")
        
        Returns:
            The prompt text or None if not found
        """
        if not self.sheets_manager:
            print(f"[DEBUG] sheets_manager is None, cannot fetch prompt for assignment {assignment_id}")
            return None
            
        try:
            # Get the assignment record using the fetch method
            print(f"[DEBUG] Fetching assignment {assignment_id} for prompt type {prompt_type}")
            assignment = self.sheets_manager.assignments.fetch(assignment_id)
            
            if not assignment:
                print(f"[DEBUG] No assignment found for ID {assignment_id}")
                return None
            
            # Map prompt type to column name
            column_map = {
                "grading": "GradingPrompt",
                "conversation": "ConversationPrompt"
            }
            
            column_name = column_map.get(prompt_type)
            if not column_name:
                print(f"[DEBUG] Invalid prompt type: {prompt_type}")
                return None
            
            prompt = assignment.get(column_name, "").strip()
            if prompt:
                print(f"[DEBUG] Successfully loaded {prompt_type} prompt (length: {len(prompt)})")
            else:
                print(f"[DEBUG] No prompt found in column {column_name} for assignment {assignment_id}")
            return prompt if prompt else None
            
        except Exception as e:
            st.error(f"Error loading prompt from assignments sheet: {e}")
            print(f"[DEBUG] Exception loading prompt: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_prompt_cached(self, assignment_id: str, prompt_type: str) -> Optional[str]:
        """Get a prompt from assignments sheet with caching.
        Only caches successful prompt fetches, not None values."""
        cache_key = f"assignment_{assignment_id}_{prompt_type}"
        
        # Check if we have a cached value
        if cache_key in self._prompts_cache:
            cached_value = self._prompts_cache[cache_key]
            print(f"[DEBUG] Using cached {prompt_type} prompt for assignment {assignment_id} (cached length: {len(cached_value) if cached_value else 'None'})")
            return cached_value
        
        # Fetch fresh prompt
        print(f"[DEBUG] Cache miss - fetching fresh {prompt_type} prompt for assignment {assignment_id}")
        prompt = self.get_prompt_from_assignment(assignment_id, prompt_type)
        
        # Only cache if we successfully got a prompt (not None or empty)
        if prompt:
            self._prompts_cache[cache_key] = prompt
            print(f"[DEBUG] Cached {prompt_type} prompt for assignment {assignment_id}")
        else:
            print(f"[DEBUG] Not caching empty/None prompt for {prompt_type} assignment {assignment_id}")
        
        return prompt

# Example usage and prompt templates
def get_default_prompts() -> Dict[str, str]:
    """Fallback prompts if Google Sheets prompts are unavailable."""
    return {
        "grading_prompt": """You are the Devil's Advocate in a debate simulation. Your role is to argue AGAINST the user's position, no matter what stance they take.

Debate Topic: {debate_topic}

User's Argument: {user_argument}

Your task:
1. Analyze the user's argument carefully
2. Identify the weaknesses, flaws, or gaps in their reasoning  
3. Generate a strong counter-argument that opposes their position
4. Challenge their points with logic, evidence, and alternative perspectives
5. Be intellectually rigorous but maintain a conversational tone

Respond directly to what the user said. Counter their specific points and make them defend their stance.

Student ID: {student_id}
Execution ID: {execution_id}

Current Student Answers:
Q1: {q1}
Q2: {q2}
Q3: {q3}

Question 1 Context History:
{q1_context}

Question 2 Context History:
{q2_context}

Question 3 Context History:
{q3_context}

Previous Conversations:
{conversation_context}

Please provide your evaluation in the following JSON format:
{{
    "execution_id": "{execution_id}",
    "student_id": "{student_id}",
    "score1": <score from 1-10>,
    "score2": <score from 1-10>,
    "score3": <score from 1-10>,
    "feedback1": "<detailed feedback for Q1>",
    "feedback2": "<detailed feedback for Q2>",
    "feedback3": "<detailed feedback for Q3>"
}}

Be thoughtful in your evaluation. Consider clarity, depth of understanding, and relevance to the questions. Use the context history to provide more personalized and relevant feedback.""",
        
        "evaluation_prompt": """You are an AI teaching assistant providing improved feedback. Review the previous grading and provide enhanced feedback.

Previous Grading Results:
{previous_grading}

Please provide improved feedback in the following JSON format:
{{
    "execution_id": "{execution_id}",
    "student_id": "{student_id}",
    "new_score1": <improved score from 1-10>,
    "new_score2": <improved score from 1-10>,
    "new_score3": <improved score from 1-10>,
    "new_feedback1": "<enhanced feedback for Q1>",
    "new_feedback2": "<enhanced feedback for Q2>",
    "new_feedback3": "<enhanced feedback for Q3>"
}}

Provide more detailed, constructive feedback that will help the student improve.""",
        
        "conversation_prompt": """You are the Judge in a debate simulation. You use an evidence-based scoring system to determine winners.

CRITICAL: Output ONLY valid JSON (no text before/after):

{{
  "decision_made": true/false,
  "winner": "user"/"devils_advocate"/null,
  "reasoning": "Your natural explanation",
  "follow_up_question": "Question text?"/null,
  "question_for": "user"/"devils_advocate"/null
}}

EVIDENCE-BASED EVALUATION:

1. Based on the debate topic, mentally identify 10 valid claims/evidence points for each side (PRO and COUNTER positions)
2. Review what each debater has presented
3. Count how many well-supported valid claims each side has made (claims that match your list or are equivalent in strength)
4. DECISION RULE: If either side presents 3+ solid, well-supported claims → THEY WIN
5. If neither has 3 yet → Ask a follow-up question to help elicit more evidence

YOUR MINDSET:

You're actively being convinced through evidence. Express genuine uncertainty:
- "I'm not fully convinced yet because neither side has provided enough concrete evidence."
- "The User made a fair point about X, but I need to hear more specifics."
- "That claim needs support - can you back that up with examples?"

Be direct and human, not robotic:
- DON'T say: "Upon careful consideration of the arguments presented..."
- DO say: "Look, both sides have decent points, but I need actual evidence here."

FOLLOW-UP QUESTIONS:

ALWAYS phrase as actual questions (ending with ?):
- "Can you provide specific examples of how regulation has stifled innovation in practice?"
- "What concrete safety incidents are you worried about, and how would regulation prevent them?"
- "You mentioned trust - how exactly does regulation build trust compared to industry self-regulation?"

Be specific about what evidence you need to reach the 3-claim threshold.

REASONING FIELD:

Write 2-3 conversational sentences max:
- When declaring winner: "The Devil's Advocate hit the threshold with solid points about safety risks, trust-building, and historical precedents. The User kept asserting regulation kills progress but didn't back it up with concrete examples."
- When asking follow-up: "Neither side has given me enough solid evidence yet. I need specifics, not abstract claims."

Be fair, decisive, and guide the debate toward substantive evidence-based arguments."""
    } 
# ===========================
# Main Application (from da_source_app.py)
# ===========================

# --- Custom CSS for Modern Look ---
st.markdown('''
    <style>
    body {
        background-color: #f7f9fa;
    }
    .main {
        background-color: #fff;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        padding: 2rem 2rem 1rem 2rem;
        margin-top: 1.5rem;
        color: #222;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
    }
    .question-card {
        background: rgba(200,210,220,0.18) !important;
        border-radius: 10px;
        padding: 0.7rem 1rem 0.7rem 1rem;
        margin-bottom: 0.7rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
        color: #fff !important;
        font-size: 1.08rem;
        font-weight: 400;
    }
    .question-card, .question-card * {
        color: #fff !important;
    }
    .question-card h3, .feedback-card h3, .main h1, .main h2, .main h3 {
        color: #fff !important;
        font-size: 1.25rem !important;
        margin-bottom: 0.3rem !important;
        margin-top: 0.3rem !important;
        font-weight: 400 !important;
    }
    .main h1 {
        font-size: 1.7rem !important;
    }
    .main h2 {
        font-size: 1.3rem !important;
    }
    .main h3 {
        font-size: 1.1rem !important;
    }
    .score-high { color: #2ecc40; font-weight: bold; }
    .score-mid { color: #f1c40f; font-weight: bold; }
    .score-low { color: #e74c3c; font-weight: bold; }
    .stButton>button {
        border-radius: 8px;
        padding: 0.5rem 1.2rem;
        font-size: 1.1rem;
        background: linear-gradient(90deg, #425066 0%, #7a8ca3 100%) !important;
        color: #fff !important;
        border: none;
        margin-top: 0.5rem;
    }
    .stTextInput>div>input, .stTextArea>div>textarea {
        border-radius: 8px;
        border: 1px solid #d0d7de;
        background: #f7f9fa;
        font-size: 1.1rem;
    }
    .stTextInput>div>input:focus, .stTextArea>div>textarea:focus {
        border: 1.5px solid #4f8cff;
        background: #fff;
    }
    .stAlert {
        border-radius: 8px;
    }
    .footer {
        margin-top: 2.5rem;
        text-align: center;
        color: #888;
        font-size: 0.95rem;
    }
    /* --- FORCE FEEDBACK CARD COLOR OVERRIDE --- */
    .feedback-card, .feedback-card * {
        background: rgba(60, 180, 90, 0.10) !important;
        color: #fff !important;
    }
    </style>
''', unsafe_allow_html=True)

# --- Header/Banner ---
st.markdown("""
<div style='display: flex; align-items: center; justify-content: center; margin-bottom: 1.2rem;'>
    <img src='https://img.icons8.com/color/96/000000/artificial-intelligence.png' style='height: 48px; margin-right: 14px;'>
    <div>
        <h1 style='margin-bottom: 0.1rem; font-size:1.6rem;'>Dynamic AI Assignment</h1>
    </div>
</div>
""", unsafe_allow_html=True)

# Add this helper at the top (after imports)
def rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()

# Get credentials and initialize prompt manager
OPENAI_API_KEY = get_openai_api_key()
GCP_CREDENTIALS = get_gcp_credentials()

# Get Gemini API key if using Gemini
GEMINI_API_KEY = None
if LLM_PROVIDER == "gemini":
    try:
        GEMINI_API_KEY = get_gemini_api_key()
    except:
        st.warning("Gemini API key not found. Falling back to OpenAI.")
        LLM_PROVIDER = "openai"

# Prompt manager will be initialized after sheets are set up
prompt_manager = None

# Generic Google Sheets wrapper with caching
class Sheet:
    def __init__(self, client, title: str, headers: list[str]):
        self.headers = headers
        self._cache = {}
        self._cache_timestamp = 0
        self._cache_ttl = 30  # Cache for 30 seconds
        ss = client.open(SPREADSHEET_NAME)
        try:
            self.ws = ss.worksheet(title)
        except gspread.exceptions.WorksheetNotFound:
            self.ws = ss.add_worksheet(title=title, rows="1000", cols=str(len(headers)))
            self.ws.append_row(headers)

    def get_all(self) -> list[dict]:
        """Get all records with caching."""
        current_time = time.time()
        if (current_time - self._cache_timestamp) < self._cache_ttl and self._cache:
            return self._cache
        
        # Fetch fresh data
        self._cache = self.ws.get_all_records()
        self._cache_timestamp = current_time
        return self._cache

    def is_duplicate(self, data: dict[str, any]) -> bool:
        # Only check for duplicates based on unique keys (e.g., execution_id, assignment_id, student_id)
        # If all keys in headers are present and match, consider it a duplicate
        for rec in self.get_all():
            if all(str(rec.get(h, "")) == str(data.get(h, "")) for h in self.headers if h in data):
                return True
        return False

    def append_row(self, data: dict[str, Any]) -> None:
        if not self.is_duplicate(data):
            row = [data.get(h, "") for h in self.headers]
            self.ws.append_row(row)
            # Invalidate cache after write
            self._cache = {}
            self._cache_timestamp = 0

# Specific sheet classes
class AssignmentsSheet(Sheet):
    def __init__(self, client):
        super().__init__(client, "assignments", [
            "date", "assignment_id", "Question1", "Question2", "Question3",
            "GradingPrompt", "ConversationPrompt"
        ])

    def fetch(self, assignment_id: str) -> dict[str, Any]:
        key = str(assignment_id).strip().lower()
        for rec in self.get_all():
            if str(rec.get("assignment_id", "")).strip().lower() == key:
                return rec
        return {}
    
    def fetch_debate_assignment(self, assignment_id: str) -> dict[str, Any]:
        """Fetch a debate assignment (ends with _da)"""
        key = str(assignment_id).strip().lower()
        for rec in self.get_all():
            rec_id = str(rec.get("assignment_id", "")).strip().lower()
            if rec_id == key and rec_id.endswith("_da"):
                return rec
        return {}
    
    def get_all_debate_assignments(self) -> list[dict[str, Any]]:
        """Get all assignments that end with _da"""
        all_assignments = self.get_all()
        debate_assignments = []
        for rec in all_assignments:
            assignment_id = str(rec.get("assignment_id", "")).strip().lower()
            if assignment_id.endswith("_da"):
                debate_assignments.append(rec)
        return debate_assignments

class StudentAssignmentsSheet(Sheet):
    def __init__(self, client):
        super().__init__(client, "student_assignments", [
            "student_id", "student_first_name", "student_last_name",
            "assignment_id", "assignment_due", "started"
        ])

    def fetch_current(self, student_id: str) -> dict[str, Any]:
        """Fetch current assignment for a student, prioritizing _da assignments"""
        sid = student_id.strip()
        today = datetime.date.today()
        debate_assignment = None
        regular_assignment = None
        
        for rec in self.get_all():
            if str(rec.get("student_id", "")).strip() == sid:
                due_str = str(rec.get("assignment_due", "")).strip()
                due = None
                for fmt in ("%Y-%m-%d", "%m/%d/%Y"):  # ISO or US
                    try:
                        due = datetime.datetime.strptime(due_str, fmt).date()
                        break
                    except ValueError:
                        due = None
                if due and due >= today:
                    assignment_id = str(rec.get("assignment_id", "")).strip().lower()
                    if assignment_id.endswith("_da"):
                        # Prioritize debate assignments
                        debate_assignment = rec
                    else:
                        regular_assignment = rec
        
        # Return debate assignment if found, otherwise regular assignment
        return debate_assignment if debate_assignment else (regular_assignment if regular_assignment else {})

class DataSheets:
    def __init__(self, creds: dict):
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
        creds_obj = Credentials.from_service_account_info(creds, scopes=scopes)
        client = gspread.authorize(creds_obj)
        self.assignments = AssignmentsSheet(client)
        self.student_assignments = StudentAssignmentsSheet(client)
        self.answers = Sheet(client, "student_answers", [
            "execution_id", "assignment_id", "student_id",
            "q1_answer", "q2_answer", "q3_answer", "timestamp"
        ])
        self.grading = Sheet(client, "feedback+grading", [
            "execution_id", "assignment_id", "student_id",
            "feedback1", "feedback2", "feedback3",
            "score1", "score2", "score3", "timestamp"
        ])
        self.evaluation = Sheet(client, "feedback_evaluation", [
            "execution_id", "assignment_id", "student_id",
            "new_feedback1", "new_feedback2", "new_feedback3",
            "new_score1", "new_score2", "new_score3", "timestamp"
        ])
        self.conversations = Sheet(client, "conversations", [
            "execution_id", "assignment_id", "student_id",
            "user_msg", "agent_msg", "timestamp"
        ])
    
    def get_latest_answers(self, student_id: str, assignment_id: str) -> dict[str, Any]:
        """Get the most recent answers for a student and assignment."""
        sid = student_id.strip()
        aid = assignment_id.strip()
        latest_record = {}
        latest_timestamp = None
        
        for rec in self.answers.get_all():
            if (str(rec.get("student_id", "")).strip() == sid and 
                str(rec.get("assignment_id", "")).strip() == aid):
                timestamp_str = rec.get("timestamp", "")
                if timestamp_str and (latest_timestamp is None or timestamp_str > latest_timestamp):
                    latest_timestamp = timestamp_str
                    latest_record = rec
        
        return latest_record
    
    def get_all_answers_for_memory(self, student_id: str, assignment_id: str) -> List[dict[str, Any]]:
        """Get all answers for a student and assignment for memory loading."""
        sid = student_id.strip()
        aid = assignment_id.strip()
        all_answers = []
        
        for rec in self.answers.get_all():
            if (str(rec.get("student_id", "")).strip() == sid and 
                str(rec.get("assignment_id", "")).strip() == aid):
                all_answers.append(rec)
        
        # Sort by timestamp to maintain chronological order
        all_answers.sort(key=lambda x: x.get("timestamp", ""))
        return all_answers
    
    def get_latest_grading(self, student_id: str, assignment_id: str) -> dict[str, Any]:
        """Get the most recent grading for a student and assignment."""
        sid = student_id.strip()
        aid = assignment_id.strip()
        latest_record = {}
        latest_timestamp = None
        
        for rec in self.grading.get_all():
            if (str(rec.get("student_id", "")).strip() == sid and 
                str(rec.get("assignment_id", "")).strip() == aid):
                timestamp_str = rec.get("timestamp", "")
                if timestamp_str and (latest_timestamp is None or timestamp_str > latest_timestamp):
                    latest_timestamp = timestamp_str
                    latest_record = rec
        
        return latest_record
    
    def get_all_grading_for_memory(self, student_id: str, assignment_id: str) -> List[dict[str, Any]]:
        """Get all grading records for a student and assignment for memory loading."""
        sid = student_id.strip()
        aid = assignment_id.strip()
        all_grading = []
        
        for rec in self.grading.get_all():
            if (str(rec.get("student_id", "")).strip() == sid and 
                str(rec.get("assignment_id", "")).strip() == aid):
                all_grading.append(rec)
        
        # Sort by timestamp to maintain chronological order
        all_grading.sort(key=lambda x: x.get("timestamp", ""))
        return all_grading
    
    def get_latest_conversation(self, student_id: str, assignment_id: str) -> dict[str, Any]:
        """Get the most recent conversation for a student and assignment."""
        sid = student_id.strip()
        aid = assignment_id.strip()
        latest_record = {}
        latest_timestamp = None
        
        for rec in self.conversations.get_all():
            if (str(rec.get("student_id", "")).strip() == sid and 
                str(rec.get("assignment_id", "")).strip() == aid):
                timestamp_str = rec.get("timestamp", "")
                if timestamp_str and (latest_timestamp is None or timestamp_str > latest_timestamp):
                    latest_timestamp = timestamp_str
                    latest_record = rec
        
        return latest_record
    
    def get_all_conversations_for_memory(self, student_id: str, assignment_id: str) -> List[dict[str, Any]]:
        """Get all conversations for a student and assignment for memory loading."""
        sid = student_id.strip()
        aid = assignment_id.strip()
        all_conversations = []
        
        for rec in self.conversations.get_all():
            if (str(rec.get("student_id", "")).strip() == sid and 
                str(rec.get("assignment_id", "")).strip() == aid):
                all_conversations.append(rec)
        
        # Sort by timestamp to maintain chronological order
        all_conversations.sort(key=lambda x: x.get("timestamp", ""))
        return all_conversations
    
    def update_started_status(self, student_id: str, assignment_id: str, started: str):
        """Update the started status for a student assignment."""
        sid = student_id.strip()
        aid = assignment_id.strip()
        
        # Find the row to update
        for i, rec in enumerate(self.student_assignments.get_all()):
            if (str(rec.get("student_id", "")).strip() == sid and 
                str(rec.get("assignment_id", "")).strip() == aid):
                # Update the started field
                row_num = i + 2  # +2 because sheets are 1-indexed and we have a header row
                self.student_assignments.ws.update_cell(row_num, 6, started)  # Column 6 is "started"
                break

# Initialize sheets
@st.cache_resource(show_spinner="Loading...")
def get_sheets() -> DataSheets:
    return DataSheets(GCP_CREDENTIALS)

sheets = get_sheets()

# Initialize prompt manager with sheets reference
@st.cache_resource
def get_prompt_manager(_sheets):
    """Initialize prompt manager with sheets reference. 
    The _sheets parameter ensures it's recreated if sheets changes."""
    pm = PromptManager(GCP_CREDENTIALS, sheets_manager=_sheets)
    return pm

prompt_manager = get_prompt_manager(sheets)

# --- OLD CONTEXT CACHE SYSTEM (COMMENTED OUT) ---
# class ContextCache:
#     """Manages context caches for questions and conversations."""
#     
#     def __init__(self):
#         # Initialize question caches with empty content
#         self.question_caches = {
#             'q1': '',
#             'q2': '', 
#             'q3': ''
#         }
#         self.conversation_cache = ''
#         self.question_counters = {'q1': 0, 'q2': 0, 'q3': 0}
#         self.conversation_counter = 0
#     
#     def initialize_question_cache(self, question_num: str, question_text: str):
#         """Initialize a question cache with the question text."""
#         self.question_caches[f'q{question_num}'] = f"<question_text>{question_text}</question_text>"
#         self.question_counters[f'q{question_num}'] = 0
#     
#     def add_response_and_feedback(self, question_num: str, response: str, feedback: str, score: str = ""):
#         """Add student response and feedback to question cache."""
#         q_key = f'q{question_num}'
#         self.question_counters[q_key] += 1
#         counter = self.question_counters[q_key]
#         
#         score_text = f" (Score: {score})" if score else ""
#         new_content = f"<response_{counter}>{response}</response_{counter}><feedback_{counter}>{feedback}{score_text}</feedback_{counter}>"
#         self.question_caches[q_key] += new_content
#     
#     def add_conversation(self, question: str, response: str):
#         """Add conversation Q&A to conversation cache."""
#         self.conversation_counter += 1
#         counter = self.conversation_counter
#         new_content = f"<conversation_question_{counter}>{question}</conversation_question_{counter}><conversation_response_{counter}>{response}</conversation_response_{counter}>"
#         self.conversation_cache += new_content
#     
#     def get_question_context(self, question_num: str) -> str:
#         """Get the full context for a specific question."""
#         return self.question_caches.get(f'q{question_num}', '')
#     
#     def get_conversation_context(self) -> str:
#         """Get the full conversation context."""
#         return self.conversation_cache
#     
#     def clear_all(self):
#         """Clear all caches."""
#         self.question_caches = {'q1': '', 'q2': '', 'q3': ''}
#         self.conversation_cache = ''
#         self.question_counters = {'q1': 0, 'q2': 0, 'q3': 0}
#         self.conversation_counter = 0

# Initialize context cache
# @st.cache_resource
# def get_context_cache():
#     return ContextCache()

# context_cache = get_context_cache()

# --- NEW LANGGRAPH MEMORY MANAGEMENT SYSTEM ---
class AssignmentState(MessagesState):
    """Enhanced state using LangGraph's MessagesState for automatic memory management."""
    
    # Core assignment data
    execution_id: str
    student_id: str
    assignment_id: str
    
    # Questions and answers
    questions: Dict[str, str]  # {"q1": "What is...", "q2": "Explain...", "q3": "How does..."}
    answers: Dict[str, str]   # {"q1": "Student answer...", "q2": "...", "q3": "..."}
    
    # Grading results
    scores: Dict[str, int]    # {"q1": 8, "q2": 7, "q3": 9}
    feedback: Dict[str, str]  # {"q1": "Good work...", "q2": "Needs improvement...", "q3": "Excellent..."}
    
    # Session metadata
    session_metadata: Dict[str, Any]
    conversation_ready: bool
    
    # Legacy compatibility fields (for smooth transition)
    question_contexts: Dict[str, str]  # Maintains question-specific context for backward compatibility

# Initialize memory system
@st.cache_resource
def get_memory_system():
    """Initialize LangGraph memory system with persistence."""
    return MemorySaver()

memory_system = get_memory_system()

# Global state manager for the assignment session
class AssignmentMemoryManager:
    """Manages assignment state using LangGraph's memory system."""
    
    def __init__(self):
        self.memory = memory_system
        self.current_state = None
    
    def initialize_assignment_session(self, exec_id: str, sid: str, aid: str, questions: Dict[str, str]) -> AssignmentState:
        """Initialize a new assignment session with questions."""
        state = {
            "messages": [SystemMessage(content=f"Assignment session for student {sid}")],
            "execution_id": exec_id,
            "student_id": sid,
            "assignment_id": aid,
            "questions": questions,
            "answers": {"q1": "", "q2": "", "q3": ""},
            "scores": {},
            "feedback": {},
            "session_metadata": {
                "start_time": datetime.datetime.now().isoformat(),
                "assignment_id": aid
            },
            "conversation_ready": True,
            "question_contexts": {"q1": "", "q2": "", "q3": ""}
        }
        self.current_state = state
        return state
    
    def add_student_answer(self, question_num: str, answer: str):
        """Add a student's answer to the current state."""
        if self.current_state:
            self.current_state["answers"][f"q{question_num}"] = answer
            # Add to conversation history
            self.current_state["messages"].append(HumanMessage(content=f"Student answered Q{question_num}: {answer[:100]}..."))
    
    def add_grading_result(self, question_num: str, score: int, feedback: str):
        """Add grading results for a question."""
        if self.current_state:
            self.current_state["scores"][f"q{question_num}"] = score
            self.current_state["feedback"][f"q{question_num}"] = feedback
            
            # Add to conversation history
            self.current_state["messages"].append(AIMessage(content=f"Q{question_num} graded: {score}/10 - {feedback[:100]}..."))
            
            # Update question context for backward compatibility
            q_key = f"q{question_num}"
            if q_key not in self.current_state["question_contexts"]:
                self.current_state["question_contexts"][q_key] = ""
            
            # Add response and feedback to question context (maintaining backward compatibility)
            counter = len([msg for msg in self.current_state["messages"] if f"Q{question_num}" in msg.content])
            score_text = f" (Score: {score})"
            new_content = f"<response_{counter}>{self.current_state['answers'][f'q{question_num}']}</response_{counter}><feedback_{counter}>{feedback}{score_text}</feedback_{counter}>"
            self.current_state["question_contexts"][q_key] += new_content
    
    def add_conversation(self, question: str, response: str):
        """Add a conversation exchange to the memory."""
        if self.current_state:
            # Add to conversation history
            self.current_state["messages"].append(HumanMessage(content=question))
            self.current_state["messages"].append(AIMessage(content=response))
    
    def get_question_context(self, question_num: str) -> str:
        """Get question-specific context (backward compatibility)."""
        if self.current_state:
            return self.current_state["question_contexts"].get(f"q{question_num}", "")
        return ""
    
    def get_conversation_context(self) -> str:
        """Get conversation context from messages (backward compatibility)."""
        if self.current_state:
            # Convert messages to conversation context format for backward compatibility
            conversation_parts = []
            for msg in self.current_state["messages"]:
                if isinstance(msg, HumanMessage):
                    conversation_parts.append(f"Student: {msg.content}")
                elif isinstance(msg, AIMessage):
                    conversation_parts.append(f"AI: {msg.content}")
            return "\n".join(conversation_parts)
        return ""
    
    def get_full_conversation_history(self) -> List:
        """Get the full conversation history as messages."""
        if self.current_state:
            return self.current_state["messages"]
        return []
    
    def clear_session(self):
        """Clear the current session."""
        self.current_state = None
    
    def get_current_state(self) -> AssignmentState:
        """Get the current assignment state."""
        return self.current_state

# Initialize assignment memory manager
@st.cache_resource
def get_assignment_memory_manager():
    return AssignmentMemoryManager()

assignment_memory = get_assignment_memory_manager()

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_previous_session_data(student_id: str, assignment_id: str) -> tuple[Dict[str, str], Dict[str, Any], str]:
    """
    Load previous session data for a student and assignment.
    Returns: (previous_answers, previous_feedback, latest_conversation_response)
    """
    try:
        print(f"[SESSION RESTORE] Loading data for student {student_id}, assignment {assignment_id}")
        
        # Get all data in one go to minimize API calls
        all_answers = sheets.get_all_answers_for_memory(student_id, assignment_id)
        all_grading = sheets.get_all_grading_for_memory(student_id, assignment_id)
        all_conversations = sheets.get_all_conversations_for_memory(student_id, assignment_id)
        
        # Find the most recent answers record
        latest_answers = {}
        if all_answers:
            latest_answers = max(all_answers, key=lambda x: x.get("timestamp", ""))
            print(f"[DEBUG] Latest answers record: {latest_answers}")
        
        # Find the matching feedback record with the same execution_id
        matching_feedback = {}
        if latest_answers and all_grading:
            execution_id = latest_answers.get("execution_id")
            print(f"[DEBUG] Looking for feedback with execution_id: {execution_id}")
            
            for grading_record in all_grading:
                if grading_record.get("execution_id") == execution_id:
                    matching_feedback = grading_record
                    print(f"[DEBUG] Found matching feedback record: {matching_feedback}")
                    break
        
        # Find the most recent conversation record
        latest_conversation = {}
        if all_conversations:
            latest_conversation = max(all_conversations, key=lambda x: x.get("timestamp", ""))
        
        # Prepare previous answers for UI
        previous_answers = {}
        for i in range(1, 4):
            answer_key = f"q{i}_answer"  # Use correct column name from Google Sheet
            ui_key = f"q{i}"  # Use q1, q2, q3 for UI
            if answer_key in latest_answers:
                previous_answers[ui_key] = latest_answers[answer_key]
                print(f"[DEBUG] Found {answer_key}: {latest_answers[answer_key][:50]}..." if latest_answers[answer_key] else f"[DEBUG] Found {answer_key}: (empty)")
            else:
                print(f"[DEBUG] Missing {answer_key} in latest_answers")
        
        # Get latest conversation response for UI
        latest_conversation_response = latest_conversation.get("agent_msg", "") if latest_conversation else ""
        
        print(f"[SESSION RESTORE] Found {len(all_answers)} answer records, {len(all_grading)} grading records, {len(all_conversations)} conversation records")
        print(f"[SESSION RESTORE] Previous answers: {previous_answers}")
        print(f"[SESSION RESTORE] Matching feedback: {matching_feedback}")
        print(f"[SESSION RESTORE] Latest conversation response: {latest_conversation_response[:100]}..." if latest_conversation_response else "[SESSION RESTORE] No conversation response")
        
        return previous_answers, matching_feedback, latest_conversation_response
        
    except Exception as e:
        print(f"[ERROR] Failed to load previous session data: {e}")
        return {}, {}, ""

def load_session_data_into_memory(student_id: str, assignment_id: str):
    """
    Load all session data into memory system (separate from UI data loading).
    This is called only once per session.
    """
    try:
        print(f"[MEMORY LOAD] Loading data into memory for student {student_id}, assignment {assignment_id}")
        
        # Get all data for memory loading
        all_answers = sheets.get_all_answers_for_memory(student_id, assignment_id)
        all_grading = sheets.get_all_grading_for_memory(student_id, assignment_id)
        all_conversations = sheets.get_all_conversations_for_memory(student_id, assignment_id)
        
        # Load all answers and feedback into memory
        for answer_record in all_answers:
            for i in range(1, 4):
                answer_key = f"q{i}_answer"  # Use correct column name from Google Sheet
                if answer_key in answer_record and answer_record[answer_key]:
                    assignment_memory.add_student_answer(str(i), answer_record[answer_key])
        
        for grading_record in all_grading:
            for i in range(1, 4):
                score_key = f"score{i}"
                feedback_key = f"feedback{i}"
                if score_key in grading_record and feedback_key in grading_record:
                    score = grading_record[score_key]
                    feedback = grading_record[feedback_key]
                    if score and feedback:
                        assignment_memory.add_grading_result(str(i), int(score) if str(score).isdigit() else 0, feedback)
        
        # Load all conversations into memory
        for conv_record in all_conversations:
            user_msg = conv_record.get("user_msg", "")
            agent_msg = conv_record.get("agent_msg", "")
            if user_msg and agent_msg:
                assignment_memory.add_conversation(user_msg, agent_msg)
        
        print(f"[MEMORY LOAD] Loaded {len(all_answers)} answer records, {len(all_grading)} grading records, {len(all_conversations)} conversation records into memory")
        
    except Exception as e:
        print(f"[ERROR] Failed to load session data into memory: {e}")

# Backward compatibility wrapper for existing code
class ContextCache:
    """Backward compatibility wrapper that uses the new memory system."""
    
    def __init__(self):
        self.memory_manager = assignment_memory
    
    def initialize_question_cache(self, question_num: str, question_text: str):
        """Initialize question cache (maintains backward compatibility)."""
        if self.memory_manager.current_state:
            self.memory_manager.current_state["question_contexts"][f"q{question_num}"] = f"<question_text>{question_text}</question_text>"
    
    def add_response_and_feedback(self, question_num: str, response: str, feedback: str, score: str = ""):
        """Add response and feedback (maintains backward compatibility)."""
        # Add student's response if provided
        if response:
            self.memory_manager.add_student_answer(question_num, response)
        # Add feedback if provided
        if feedback:
            self.memory_manager.add_grading_result(question_num, int(score) if score else 0, feedback)
    
    def add_conversation(self, question: str, response: str):
        """Add conversation (maintains backward compatibility)."""
        self.memory_manager.add_conversation(question, response)
    
    def get_question_context(self, question_num: str) -> str:
        """Get question context (maintains backward compatibility)."""
        return self.memory_manager.get_question_context(question_num)
    
    def get_conversation_context(self) -> str:
        """Get conversation context (maintains backward compatibility)."""
        return self.memory_manager.get_conversation_context()
    
    def clear_all(self):
        """Clear all caches (maintains backward compatibility)."""
        self.memory_manager.clear_session()

# Initialize context cache for backward compatibility
@st.cache_resource
def get_context_cache():
    return ContextCache()

context_cache = get_context_cache()

# --- Simple Background Writer System ---
class SimpleBackgroundWriter:
    """Simple background writer for Google Sheets operations."""
    
    def __init__(self, sheets_instance):
        self.sheets = sheets_instance
        self.active_threads = []
    
    def write_async(self, operation_type: str, data: dict):
        """Start a background thread to write data to sheets."""
        def write_worker():
            try:
                if operation_type == 'answers':
                    self.sheets.answers.append_row(data)
                elif operation_type == 'grading':
                    self.sheets.grading.append_row(data)
                elif operation_type == 'evaluation':
                    self.sheets.evaluation.append_row(data)
                elif operation_type == 'conversations':
                    self.sheets.conversations.append_row(data)
                print(f"[DEBUG] Successfully wrote {operation_type} data to sheets")
            except Exception as e:
                print(f"[ERROR] Failed to write {operation_type} data: {e}")
        
        # Start background thread
        thread = threading.Thread(target=write_worker, daemon=True)
        thread.start()
        self.active_threads.append(thread)
        
        # Clean up completed threads (keep only last 10)
        self.active_threads = [t for t in self.active_threads if t.is_alive()][-10:]
    
    def shutdown(self):
        """Wait for all active threads to complete."""
        for thread in self.active_threads:
            if thread.is_alive():
                thread.join(timeout=5)

# Initialize simple background writer
@st.cache_resource
def get_background_writer():
    return SimpleBackgroundWriter(sheets)

background_writer = get_background_writer()

# Initialize LangGraph agents with memory
@st.cache_resource
def get_devils_advocate_agent():
    """Create Devil's Advocate agent with memory."""
    if LLM_PROVIDER == "gemini" and GEMINI_API_KEY:
        llm = ChatGoogleGenerativeAI(
            model=DEFAULT_MODEL["gemini"],
            temperature=1,
            google_api_key=GEMINI_API_KEY,
            streaming=True,
            max_output_tokens=4000,
            request_timeout=60
        )
    else:
        llm = ChatOpenAI(
            model_name=DEFAULT_MODEL["openai"], 
            temperature=1,
            openai_api_key=OPENAI_API_KEY,
            streaming=True,
            max_tokens=4000,
            request_timeout=60
        )
    
    # Create agent with memory
    agent = create_react_agent(llm, tools=[], checkpointer=memory_system)
    return agent

@st.cache_resource
def get_judge_agent():
    """Create Judge agent with memory."""
    if LLM_PROVIDER == "gemini" and GEMINI_API_KEY:
        llm = ChatGoogleGenerativeAI(
            model=DEFAULT_MODEL["gemini"],
            temperature=0.7,
            google_api_key=GEMINI_API_KEY,
            streaming=True,
            max_output_tokens=4000,
            request_timeout=60
        )
    else:
        llm = ChatOpenAI(
            model_name=DEFAULT_MODEL["openai"], 
            temperature=0.7,
            openai_api_key=OPENAI_API_KEY,
            streaming=True,
            max_tokens=4000,
            request_timeout=60
        )
    
    # Create agent with memory
    agent = create_react_agent(llm, tools=[], checkpointer=memory_system)
    return agent

devils_advocate_agent = get_devils_advocate_agent()
judge_agent = get_judge_agent()

# Workflow functions

def prompt_student_id() -> Optional[str]:
    sid = st.text_input('Student ID:', key='sid')
    if not sid:
        st.info('Enter your Student ID to proceed.')
        return None
    return sid.strip()


def load_assignment(sid: str) -> Optional[dict[str, Any]]:
    rec = sheets.student_assignments.fetch_current(sid)
    if not rec:
        st.error('Invalid Student ID or no assignment due.')
        return None
    
    # Check if this is a debate assignment
    assignment_id = str(rec.get("assignment_id", "")).strip()
    if assignment_id.lower().endswith("_da"):
        print(f"[DEBUG] Loading DEBATE assignment: {assignment_id}")
        st.session_state['is_debate_mode'] = True
    else:
        print(f"[DEBUG] Loading QUIZ assignment: {assignment_id}")
        st.session_state['is_debate_mode'] = False
    
    return rec


def load_questions(aid: str) -> Optional[dict[str, Any]]:
    """Load questions for an assignment (handles both quiz and debate modes)"""
    q = sheets.assignments.fetch(aid)
    if not q:
        st.error('Assignment questions not found.')
        return None
    
    # Check if this is a debate assignment
    is_debate = aid.lower().endswith("_da")
    
    if is_debate:
        # For debate mode, we only need Question1 (the debate topic)
        print(f"[DEBUG] Loaded debate topic: {q.get('Question1', 'N/A')}")
        # Ensure Question2 and Question3 are empty for debate mode
        q['Question2'] = ''
        q['Question3'] = ''
    else:
        print(f"[DEBUG] Loaded quiz with 3 questions")
    
    return q


def record_answers(exec_id: str, sid: str, aid: str, answers: dict[str, str]) -> None:
    """Record answers using background writer for better performance."""
    data = {
        'execution_id': exec_id,
        'assignment_id': aid,
        'student_id': sid,
        'q1_answer': answers.get('q1', ''),
        'q2_answer': answers.get('q2', ''),
        'q3_answer': answers.get('q3', ''),
        'timestamp': datetime.datetime.now().isoformat(sep=' ', timespec='seconds')
    }
    # Queue for background writing instead of blocking
    background_writer.write_async('answers', data)


# Removed grade_single_question function - now using _make_single_api_call for true parallelism


def run_grading_streaming(exec_id: str, sid: str, aid: str, answers: Dict[str, str]) -> Dict[str, Any]:
    """Devil's Advocate response - generates counter-argument to user's position."""
    try:
        # Get the debate topic (stored as Question1 in assignments)
        debate_topic = ""
        try:
            assignment = sheets.assignments.fetch(aid)
            debate_topic = assignment.get('Question1', 'Unknown topic')
        except:
            debate_topic = 'Unknown topic'
        
        # Get user's current argument - THIS IS WHAT WE RESPOND TO
        user_current_argument = answers.get('q1', 'No argument provided')
        
        print(f"[DEBUG] ===== USER'S CURRENT ARGUMENT =====")
        print(f"[DEBUG] {user_current_argument}")
        print(f"[DEBUG] =======================================")
        
        # Try to get Devil's Advocate prompt from assignments sheet
        prompt_template = prompt_manager.get_prompt_cached(aid, "grading")
        
        # Fall back to default if not found in sheet
        if not prompt_template:
            print(f"[DEBUG] No prompt found in sheet for assignment {aid}, using default Devil's Advocate prompt")
            prompt_template = get_default_prompts()["grading_prompt"]
        else:
            print(f"[DEBUG] Using Devil's Advocate prompt from sheet for assignment {aid}")
        
        # Print the actual prompt template being used
        print(f"[DEBUG] ===== DEVIL'S ADVOCATE PROMPT TEMPLATE =====")
        print(f"[DEBUG] {prompt_template[:500]}..." if len(prompt_template) > 500 else f"[DEBUG] {prompt_template}")
        print(f"[DEBUG] ==============================================")
        
        # Build the message for the Devil's Advocate agent
        # System message with personality + current context
        devils_advocate_message = f"""{prompt_template}

===== DEBATE TOPIC =====
{debate_topic}

===== USER'S ARGUMENT (RESPOND TO THIS) =====
{user_current_argument}
"""
        
        print(f"[DEBUG] ===== FINAL DEVIL'S ADVOCATE MESSAGE =====")
        print(f"[DEBUG] {devils_advocate_message[:800]}..." if len(devils_advocate_message) > 800 else f"[DEBUG] {devils_advocate_message}")
        print(f"[DEBUG] ==============================================")
        
        # Generate Devil's Advocate response using LangGraph agent with memory
        with st.spinner("Devil's Advocate is formulating a counter-argument..."):
            start_time = time.time()
            
            # Create thread ID for this debate (use student_id + assignment_id for persistence)
            thread_id = f"da_{sid}_{aid}"
            
            print(f"[DEBUG] Using Devil's Advocate agent with thread_id: {thread_id}")
            
            # Invoke the agent with memory
            config = {"configurable": {"thread_id": thread_id}}
            
            response_text = ""
            try:
                # Stream response from agent
                for chunk in devils_advocate_agent.stream(
                    {"messages": [HumanMessage(content=devils_advocate_message)]},
                    config=config
                ):
                    # Extract content from agent response
                    if "agent" in chunk:
                        for message in chunk["agent"]["messages"]:
                            if hasattr(message, 'content') and message.content:
                                response_text += message.content
            except Exception as e:
                print(f"[ERROR] Agent streaming failed: {e}")
                # Fallback to invoke if streaming fails
                result = devils_advocate_agent.invoke(
                    {"messages": [HumanMessage(content=devils_advocate_message)]},
                    config=config
                )
                if "messages" in result:
                    for msg in result["messages"]:
                        if isinstance(msg, AIMessage) and msg.content:
                            response_text += msg.content
            
            response_time = time.time() - start_time
            print(f"[BENCHMARK] Devil's Advocate response completed in {response_time:.3f}s")
            print(f"[DEBUG] Devil's Advocate response length: {len(response_text)} chars")
            print(f"[DEBUG] Response preview: {response_text[:100]}...")
        
        # Format result to match expected structure (using score1/feedback1)
        result = {
            "execution_id": exec_id,
            "assignment_id": aid,
            "student_id": sid,
            "score1": 0,  # Score not used for Devil's Advocate, set to 0
            "score2": 0,
            "score3": 0,
            "feedback1": response_text,  # Devil's Advocate response goes in feedback1
            "feedback2": "",
            "feedback3": ""
        }
        
        print("[DEBUG] Devil's Advocate result:", {**result, "feedback1": f"{result['feedback1'][:100]}..."})
        return result
        
    except Exception as e:
        print(f"[ERROR] Devil's Advocate response failed: {e}")
        st.error(f"Devil's Advocate response failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "execution_id": exec_id,
            "student_id": sid,
            "assignment_id": aid,
            "score1": 0,
            "score2": 0,
            "score3": 0,
            "feedback1": f"Error generating Devil's Advocate response: {str(e)}",
            "feedback2": "",
            "feedback3": ""
        }


def _make_single_api_call(question_num: int, prompt: str) -> Dict[str, Any]:
    """Make a single API call to the configured LLM for grading with dedicated agent instance."""
    try:
        # Create a dedicated agent instance for this thread to avoid conflicts
        if LLM_PROVIDER == "gemini" and GEMINI_API_KEY:
            thread_agent = ChatGoogleGenerativeAI(
                model=DEFAULT_MODEL["gemini"],
                temperature=1,
                google_api_key=GEMINI_API_KEY,
                streaming=True,
                max_output_tokens=4000,
                request_timeout=60
            )
        else:
            thread_agent = ChatOpenAI(
                model_name=DEFAULT_MODEL["openai"], 
                temperature=1,
                openai_api_key=OPENAI_API_KEY,
                streaming=True,
                max_tokens=4000,
                request_timeout=60
            )
        
        # Use streaming for faster response
        response_text = ""
        for chunk in thread_agent.stream(prompt):
            if hasattr(chunk, 'content'):
                response_text += chunk.content
        
        response_preview = response_text[:50] + "..." if len(response_text) > 50 else response_text
        print(f"[DEBUG] Q{question_num} API response received: {response_preview}")
        
        # Parse JSON response
        result = {}
        try:
            import re
            json_match = re.search(r'\{.*?\}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = json.loads(response_text.strip())
        except json.JSONDecodeError:
            # Fallback response - clean up any JSON artifacts
            clean_feedback = response_text
            # Remove common JSON artifacts
            clean_feedback = re.sub(r'```json\s*', '', clean_feedback)
            clean_feedback = re.sub(r'```\s*', '', clean_feedback)
            clean_feedback = re.sub(r'\{[^}]*\}', '', clean_feedback)  # Remove JSON objects
            clean_feedback = re.sub(r'"[^"]*":\s*"[^"]*"', '', clean_feedback)  # Remove key-value pairs
            clean_feedback = clean_feedback.strip()
            
            result = {
                f"score{question_num}": 5,
                f"feedback{question_num}": clean_feedback[:300] + "..." if len(clean_feedback) > 300 else clean_feedback if clean_feedback else "No feedback available"
            }
        
        return result
        
    except Exception as e:
        print(f"[ERROR] Q{question_num} API call failed: {e}")
        return {
            f"score{question_num}": 5,
            f"feedback{question_num}": f"Error grading question {question_num}: {str(e)}"
        }


def run_grading(exec_id: str, sid: str, aid: str, answers: Dict[str, str]) -> Dict[str, Any]:
    """Legacy function - now calls the optimized version."""
    return run_grading_streaming(exec_id, sid, aid, answers)


def run_evaluation_streaming(grade_res: Dict[str, Any]) -> Dict[str, Any]:
    """Optimized evaluation with streaming."""
    try:
        # Try to get prompt from Google Docs first
        doc_id = PROMPT_DOC_IDS.get("evaluation")
        prompt_name = PROMPT_NAMES.get("evaluation")
        
        if doc_id and doc_id != "YOUR_EVALUATION_PROMPT_DOC_ID":
            prompt_template = prompt_manager.get_cached_prompt(doc_id, prompt_name)
        else:
            prompt_template = get_default_prompts()["evaluation_prompt"]
        
        if not prompt_template:
            prompt_template = get_default_prompts()["evaluation_prompt"]
        
        # Format the prompt with actual data
        prompt = prompt_template.format(
            execution_id=grade_res.get('execution_id', ''),
            student_id=grade_res.get('student_id', ''),
            previous_grading=json.dumps(grade_res, indent=2)
        )

        # Use streaming for faster response
        response_text = ""
        start_time = time.time()
        with st.spinner("Evaluating feedback..."):
            for chunk in agent.stream(prompt):
                if hasattr(chunk, 'content'):
                    response_text += chunk.content
        
        eval_time = time.time() - start_time
        response_preview = response_text[:50] + "..." if len(response_text) > 50 else response_text
        print(f"[BENCHMARK] Evaluation completed in {eval_time:.3f}s")
        print(f"[DEBUG] run_evaluation_streaming LLM response: {response_preview}")
        
        # Improved JSON parsing with multiple fallback strategies
        result = {}
        try:
            import re
            # Strategy 1: Look for JSON object with simpler pattern
            json_match = re.search(r'\{.*?\}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                # Strategy 2: Look for JSON array
                json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    # Strategy 3: Try to parse the entire response as JSON
                    result = json.loads(response_text.strip())
        except json.JSONDecodeError:
            # Strategy 4: If all JSON parsing fails, try to extract fields manually
            print("[WARNING] Could not parse JSON from evaluation response, attempting manual extraction")
            result = {
                "execution_id": grade_res.get('execution_id', ''),
                "student_id": grade_res.get('student_id', ''),
                "new_score1": grade_res.get('score1', 0),
                "new_score2": grade_res.get('score2', 0),
                "new_score3": grade_res.get('score3', 0),
                "new_feedback1": grade_res.get('feedback1', ''),
                "new_feedback2": grade_res.get('feedback2', ''),
                "new_feedback3": grade_res.get('feedback3', '')
            }
            
            # Try to extract individual feedback fields from the response text
            try:
                import re
                # Look for new_feedback1, new_feedback2, new_feedback3 patterns
                for i in range(1, 4):
                    feedback_pattern = rf'"new_feedback{i}":\s*"([^"]*)"'
                    score_pattern = rf'"new_score{i}":\s*(\d+)'
                    
                    feedback_match = re.search(feedback_pattern, response_text)
                    score_match = re.search(score_pattern, response_text)
                    
                    if feedback_match:
                        result[f"new_feedback{i}"] = feedback_match.group(1)
                    if score_match:
                        result[f"new_score{i}"] = int(score_match.group(1))
                        
            except Exception as e:
                print(f"[WARNING] Manual extraction failed: {e}")
                # Keep the original feedback as fallback
        
        print("[DEBUG] run_evaluation_streaming extracted:", result)
        return result
        
    except Exception as e:
        print(f"[ERROR] run_evaluation_streaming failed: {e}")
        st.error(f"Evaluation failed: {e}")
        return {}


def run_evaluation(grade_res: Dict[str, Any]) -> Dict[str, Any]:
    """Legacy function - now calls the optimized version."""
    return run_evaluation_streaming(grade_res)


def run_conversation_streaming(exec_id: str, sid: str, user_msg: str) -> Dict[str, Any]:
    """Optimized conversation with streaming and context cache integration."""
    try:
        # Build context from cache system
        q1_context = context_cache.get_question_context('1')
        q2_context = context_cache.get_question_context('2')
        q3_context = context_cache.get_question_context('3')
        conversation_context = context_cache.get_conversation_context()
        
        # Combine all contexts
        context_str = f"Question 1 Context:\n{q1_context}\n\nQuestion 2 Context:\n{q2_context}\n\nQuestion 3 Context:\n{q3_context}\n\nConversation History:\n{conversation_context}"
        
        # Get assignment_id from session state
        aid = st.session_state.get('assignment_id', '')
        
        # Try to get prompt from assignments sheet based on assignment_id
        prompt_template = prompt_manager.get_prompt_cached(aid, "conversation")
        
        # Fall back to default if not found in sheet
        if not prompt_template:
            print(f"[DEBUG] No prompt found in sheet for assignment {aid}, using default conversation prompt")
            prompt_template = get_default_prompts()["conversation_prompt"]
        else:
            print(f"[DEBUG] Using conversation prompt from sheet for assignment {aid}")
        
        # Print the actual prompt template being used
        print(f"[DEBUG] ===== CONVERSATION PROMPT TEMPLATE =====")
        print(f"[DEBUG] {prompt_template[:500]}..." if len(prompt_template) > 500 else f"[DEBUG] {prompt_template}")
        print(f"[DEBUG] ==========================================")
        
        # Format the prompt with context-aware data
        prompt = prompt_template.format(
            context=context_str,
            user_question=user_msg
        )
        
        # Print the final formatted prompt
        print(f"[DEBUG] ===== FINAL CONVERSATION PROMPT =====")
        print(f"[DEBUG] {prompt[:500]}..." if len(prompt) > 500 else f"[DEBUG] {prompt}")
        print(f"[DEBUG] ========================================")

        # Use streaming for faster response
        response_text = ""
        start_time = time.time()
        with st.spinner("Processing your question..."):
            for chunk in agent.stream(prompt):
                if hasattr(chunk, 'content'):
                    response_text += chunk.content
        
        conv_time = time.time() - start_time
        response_preview = response_text[:50] + "..." if len(response_text) > 50 else response_text
        print(f"[BENCHMARK] Conversation completed in {conv_time:.3f}s")
        print(f"[DEBUG] run_conversation_streaming LLM response: {response_preview}")
        
        result = {
            "execution_id": exec_id,
            "student_id": sid,
            "user_msg": user_msg,
            "agent_msg": response_text,
            "timestamp": datetime.datetime.now().isoformat(sep=' ', timespec='seconds')
        }
        
        print("[DEBUG] run_conversation_streaming extracted:", result)
        return result
        
    except Exception as e:
        print(f"[ERROR] run_conversation_streaming failed: {e}")
        st.error(f"Conversation failed: {e}")
        return {
            "execution_id": exec_id,
            "student_id": sid,
            "user_msg": user_msg,
            "agent_msg": "I'm sorry, I'm having trouble processing your question right now. Please try again.",
            "timestamp": datetime.datetime.now().isoformat(sep=' ', timespec='seconds')
        }


def run_conversation(exec_id: str, sid: str, user_msg: str) -> Dict[str, Any]:
    """Legacy function - now calls the optimized streaming version."""
    return run_conversation_streaming(exec_id, sid, user_msg)


def run_judge(exec_id: str, sid: str, aid: str) -> Dict[str, Any]:
    """Judge analyzes the full debate history and makes a decision or asks follow-up questions."""
    try:
        # Get the debate topic
        assignment = sheets.assignments.fetch(aid)
        debate_topic = assignment.get('Question1', 'Unknown topic')
        
        # Build full debate history from Google Sheets
        debate_history_parts = []
        try:
            all_answers = sheets.get_all_answers_for_memory(sid, aid)
            all_grading = sheets.get_all_grading_for_memory(sid, aid)
            
            print(f"[DEBUG] Building judge history - found {len(all_answers)} answers, {len(all_grading)} grading records")
            
            # Combine answers and grading by execution_id and timestamp
            exchanges = []
            for answer_rec in all_answers:
                exec_id_key = answer_rec.get('execution_id', '')
                user_arg = answer_rec.get('q1_answer', '')
                timestamp = answer_rec.get('timestamp', '')
                if exec_id_key and user_arg:
                    exchanges.append({
                        'exec_id': exec_id_key,
                        'timestamp': timestamp,
                        'user': user_arg,
                        'da': ''
                    })
            
            # Match Devil's Advocate responses
            for grading_rec in all_grading:
                exec_id_key = grading_rec.get('execution_id', '')
                da_resp = grading_rec.get('feedback1', '')
                if exec_id_key and da_resp:
                    for exchange in exchanges:
                        if exchange['exec_id'] == exec_id_key:
                            exchange['da'] = da_resp
                            break
            
            # Sort by timestamp and build history string
            exchanges.sort(key=lambda x: x.get('timestamp', ''))
            for exchange in exchanges:
                if exchange['user']:
                    debate_history_parts.append(f"User: {exchange['user']}")
                if exchange['da']:
                    debate_history_parts.append(f"Devil's Advocate: {exchange['da']}")
            
            debate_history = "\n\n".join(debate_history_parts) if debate_history_parts else "No debate exchanges yet."
            
            print(f"[DEBUG] Built debate history with {len(exchanges)} exchanges")
            print(f"[DEBUG] History preview: {debate_history[:300]}...")
        except Exception as e:
            print(f"[ERROR] Failed to build debate history for judge: {e}")
            import traceback
            traceback.print_exc()
            debate_history = "Unable to retrieve debate history."
        
        # Get previous judge inquiries if any
        judge_history = st.session_state.get('judge_history', '')
        
        # Try to get Judge prompt from assignments sheet (using ConversationPrompt column)
        prompt_template = prompt_manager.get_prompt_cached(aid, "conversation")
        
        # Fall back to default if not found in sheet
        if not prompt_template:
            print(f"[DEBUG] No Judge prompt found in sheet for assignment {aid}, using default")
            prompt_template = get_default_prompts()["conversation_prompt"]
        else:
            print(f"[DEBUG] Using Judge prompt from sheet for assignment {aid}")
        
        print(f"[DEBUG] ===== JUDGE PROMPT TEMPLATE =====")
        print(f"[DEBUG] {prompt_template[:500]}..." if len(prompt_template) > 500 else f"[DEBUG] {prompt_template}")
        print(f"[DEBUG] ===================================")
        
        # ALWAYS append debate data to ensure Judge sees it
        judge_prompt = f"""{prompt_template}

===== DEBATE TOPIC =====
{debate_topic}

===== FULL DEBATE HISTORY =====
{debate_history}

===== PREVIOUS JUDGE INQUIRIES =====
{judge_history if judge_history else "This is the first time the Judge is being called."}

===== INSTRUCTIONS =====
Review the debate history above carefully. Based on what you see, respond ONLY with valid JSON in this exact format:

```json
{{
  "decision_made": true/false,
  "winner": "user" or "devils_advocate" or null,
  "reasoning": "Your analysis in natural, conversational language",
  "follow_up_question": "Your question" or null,
  "question_for": "user" or "devils_advocate" or null
}}
```

If declaring a winner:
- Set decision_made: true, winner: "user" or "devils_advocate"
- Write reasoning in 2-3 conversational sentences explaining why
- Set follow_up fields to null

If you need clarification:
- Set decision_made: false, winner: null
- Write brief reasoning explaining what you need (1-2 sentences)
- Provide ONE specific follow_up_question
- Specify question_for as "user" or "devils_advocate"

ONLY output the JSON. No additional text before or after.
"""
        
        print(f"[DEBUG] ===== FINAL JUDGE PROMPT =====")
        print(f"[DEBUG] {judge_prompt[:800]}..." if len(judge_prompt) > 800 else f"[DEBUG] {judge_prompt}")
        print(f"[DEBUG] ==================================")
        
        # Generate Judge's ruling using LangGraph agent with memory
        with st.spinner("⚖️ The Judge is reviewing the arguments..."):
            start_time = time.time()
            
            # Create thread ID for the judge (separate from Devil's Advocate)
            judge_thread_id = f"judge_{sid}_{aid}"
            
            print(f"[DEBUG] Using Judge agent with thread_id: {judge_thread_id}")
            
            # Invoke the agent with memory
            config = {"configurable": {"thread_id": judge_thread_id}}
            
            response_text = ""
            try:
                # Stream response from agent
                for chunk in judge_agent.stream(
                    {"messages": [HumanMessage(content=judge_prompt)]},
                    config=config
                ):
                    # Extract content from agent response
                    if "agent" in chunk:
                        for message in chunk["agent"]["messages"]:
                            if hasattr(message, 'content') and message.content:
                                response_text += message.content
            except Exception as e:
                print(f"[ERROR] Judge agent streaming failed: {e}")
                # Fallback to invoke if streaming fails
                result = judge_agent.invoke(
                    {"messages": [HumanMessage(content=judge_prompt)]},
                    config=config
                )
                if "messages" in result:
                    for msg in result["messages"]:
                        if isinstance(msg, AIMessage) and msg.content:
                            response_text += msg.content
            
            judge_time = time.time() - start_time
            print(f"[BENCHMARK] Judge ruling completed in {judge_time:.3f}s")
            print(f"[DEBUG] Judge response length: {len(response_text)} chars")
            print(f"[DEBUG] Response preview: {response_text[:100]}...")
        
        # Parse Judge's response to extract decision/follow-up
        parsed_judgment = parse_judge_response(response_text)
        
        result = {
            "execution_id": exec_id,
            "student_id": sid,
            "full_response": response_text,
            "decision_made": parsed_judgment.get("decision_made", False),
            "winner": parsed_judgment.get("winner"),
            "reasoning": parsed_judgment.get("reasoning", response_text),
            "follow_up_question": parsed_judgment.get("follow_up_question"),
            "question_for": parsed_judgment.get("question_for"),
            "timestamp": datetime.datetime.now().isoformat(sep=' ', timespec='seconds')
        }
        
        print("[DEBUG] Judge result:", {**result, "full_response": f"{result['full_response'][:100]}..."})
        return result
        
    except Exception as e:
        print(f"[ERROR] Judge ruling failed: {e}")
        st.error(f"Judge ruling failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "execution_id": exec_id,
            "student_id": sid,
            "full_response": f"Error generating Judge ruling: {str(e)}",
            "decision_made": False,
            "winner": None,
            "reasoning": "Error occurred",
            "follow_up_question": None,
            "question_for": None,
            "timestamp": datetime.datetime.now().isoformat(sep=' ', timespec='seconds')
        }


def parse_judge_response(response_text: str) -> Dict[str, Any]:
    """Parse the Judge's response to extract structured information (handles both JSON and plain text)."""
    import re
    
    result = {
        "decision_made": False,
        "winner": None,
        "reasoning": response_text,
        "follow_up_question": None,
        "question_for": None
    }
    
    # First try to parse as JSON (if the judge returns structured data)
    try:
        # Look for JSON in the response
        json_match = re.search(r'\{[^{}]*"decision_made"[^{}]*\}', response_text, re.DOTALL)
        if json_match:
            import json
            parsed = json.loads(json_match.group(0))
            print(f"[DEBUG] Successfully parsed JSON from judge response")
            return {
                "decision_made": parsed.get("decision_made", False),
                "winner": parsed.get("winner"),
                "reasoning": parsed.get("reasoning", response_text),
                "follow_up_question": parsed.get("follow_up_question"),
                "question_for": parsed.get("question_for")
            }
    except Exception as e:
        print(f"[DEBUG] Not JSON format, parsing as plain text: {e}")
    
    # Parse as plain text
    # Look for decision indicators
    if re.search(r'\b(winner|victor|stronger argument|decided in favor|ruling in favor)\b', response_text, re.IGNORECASE):
        result["decision_made"] = True
        
        # Determine winner
        if re.search(r'\b(user|student|you)\s+(win|won|has the stronger|victory|favor)\b', response_text, re.IGNORECASE):
            result["winner"] = "user"
        elif re.search(r'\b(devil\'?s? advocate|opponent)\s+(win|won|has the stronger|victory|favor)\b', response_text, re.IGNORECASE):
            result["winner"] = "devils_advocate"
    
    # Look for follow-up question indicators
    question_match = re.search(r'(question for|I ask|I need to ask|please clarify|can you explain)[:\s]+(user|devil\'?s? advocate|student|opponent)?[:\s]*(.+?)(?:\n\n|\Z)', response_text, re.IGNORECASE | re.DOTALL)
    
    if question_match:
        result["decision_made"] = False
        result["follow_up_question"] = question_match.group(3).strip() if question_match.group(3) else None
        
        # Determine who the question is for
        target = question_match.group(2)
        if target and re.search(r'\b(user|student|you)\b', target, re.IGNORECASE):
            result["question_for"] = "user"
        elif target and re.search(r'\b(devil|advocate|opponent)\b', target, re.IGNORECASE):
            result["question_for"] = "devils_advocate"
    
    print(f"[DEBUG] Parsed judge response: {result}")
    return result


# --- Legacy Context Gathering Functions Removed ---
# These functions are no longer needed since we use the ContextCache system
# which provides better performance and more structured context management

# Main UI loop

def main() -> None:
    # Handle scroll to top
    if SCROLL_AVAILABLE and st.session_state.get('scroll_to_top', False):
        scroll_to_here(0, key='scroll_to_top')
        st.session_state['scroll_to_top'] = False
    
    
    # --- Main container for app ---
    with st.container():
        sid = prompt_student_id()
        if not sid:
            return

        sa = load_assignment(sid)
        if not sa:
            return
        aid = str(sa['assignment_id'])
        
        # Store assignment info in session state
        st.session_state['assignment_id'] = aid
        is_debate_mode = st.session_state.get('is_debate_mode', False)
        
        # Show mode indicator
        if is_debate_mode:
            st.info("🎭 **Debate Mode** - Devil's Advocate Assignment")
        
        qrec = load_questions(aid)
        if not qrec:
            return

        # Ensure exec_id is generated once per assignment session
        if 'exec_id' not in st.session_state:
            st.session_state['exec_id'] = str(uuid.uuid4())
        exec_id = st.session_state['exec_id']
        
        # Check if assignment has been started and load previous session data if needed
        started_status = sa.get('started', 'FALSE').upper()
        has_previous_data = False
        previous_answers = {}
        previous_feedback = {}
        latest_conversation_response = ""
        
        # Only load previous data if the assignment has been started (started == 'TRUE')
        if started_status == 'TRUE':
            # Check if there's previous data (answers or conversations) - use cached function
            previous_answers, previous_feedback, latest_conversation_response = load_previous_session_data(sid, aid)
            
            if previous_answers or previous_feedback or latest_conversation_response:
                has_previous_data = True
                print(f"[SESSION RESTORE] Found previous data for student {sid}, assignment {aid}")
                
                # Load all previous session data into memory (only once per session)
                if not st.session_state.get('memory_loaded', False):
                    load_session_data_into_memory(sid, aid)
                    st.session_state['memory_loaded'] = True
        else:
            print(f"[SESSION] Assignment not started yet (started={started_status}) - loading fresh form")
        
        # Initialize assignment session with new memory system
        if not assignment_memory.current_state:
            questions = {
                'q1': qrec.get('Question1', ''),
                'q2': qrec.get('Question2', ''),
                'q3': qrec.get('Question3', '')
            }
            assignment_memory.initialize_assignment_session(exec_id, sid, aid, questions)
            print(f"[MEMORY] Initialized new assignment session for student {sid}")
        else:
            print(f"[MEMORY] Using existing assignment session for student {sid}")
        
        # Initialize question caches with question text (backward compatibility)
        context_cache.initialize_question_cache('1', qrec.get('Question1', ''))
        context_cache.initialize_question_cache('2', qrec.get('Question2', ''))
        context_cache.initialize_question_cache('3', qrec.get('Question3', ''))
        round_no = st.session_state.get('round', 1)
        st.markdown(f"<h3 style='margin-top:0.2rem; margin-bottom:0.7rem; font-size:1.08rem;'>📝 Round {round_no}</h3>", unsafe_allow_html=True)

        answers: dict[str, str] = {}
        # If we have previous feedback AND no current session state feedback, populate session state with it
        if previous_feedback and not st.session_state.get('feedback'):
            st.session_state['feedback'] = previous_feedback
            st.session_state['submitted'] = True
            print(f"[DEBUG] Feedback populated from previous session: {previous_feedback}")
        elif st.session_state.get('feedback'):
            print(f"[DEBUG] Using existing session state feedback: {st.session_state.get('feedback')}")
        
        # If we have a previous conversation response, populate session state with it
        if latest_conversation_response:
            st.session_state['last_conversation_response'] = latest_conversation_response
            print(f"[DEBUG] Conversation response populated from previous session: {latest_conversation_response[:100]}...")
        
        # Use session state feedback (which now contains previous feedback if available)
        fb = st.session_state.get('feedback')
        submitted = st.session_state.get('submitted', False)
        reset_counter = st.session_state.get('reset_counter', 0)
        
        # Clear retry completion flag if it exists
        if st.session_state.get('retry_completed', False):
            st.session_state['retry_completed'] = False
            print("[RETRY] Cleared retry_completed flag")
        
        print(f"[DEBUG] Main function - fb exists: {bool(fb)}, submitted: {submitted}")
        print(f"[DEBUG] Previous answers: {previous_answers}")
        print(f"[DEBUG] Previous feedback: {previous_feedback}")
        print(f"[DEBUG] Latest conversation response: {latest_conversation_response[:100]}..." if latest_conversation_response else "[DEBUG] No conversation response")
        
        # Debug: Show memory system status
        if assignment_memory.current_state:
            memory_state = assignment_memory.current_state
            print(f"[MEMORY DEBUG] Session active for student {memory_state['student_id']}")
            print(f"[MEMORY DEBUG] Messages count: {len(memory_state.get('messages', []))}")
            print(f"[MEMORY DEBUG] Scores: {memory_state.get('scores', {})}")
            print(f"[MEMORY DEBUG] Answers: {len([k for k, v in memory_state.get('answers', {}).items() if v])}/3")
        else:
            print("[MEMORY DEBUG] No active assignment session")

        # --- Debate Topic and User's Argument ---
        # Display the debate topic from Question1
        debate_topic = qrec.get('Question1', '')
        st.markdown(f"<div class='question-card' style='margin-bottom:0.8rem; font-size:1.15rem;'><b>Debate Topic:</b> {debate_topic}</div>", unsafe_allow_html=True)
        
        # Single answer box for user's argument
        key = f'a1_r{round_no}_reset{reset_counter}'
        val_key = 'q1_val'
        
        # If we have previous answer AND no current session state value, populate session state
        if previous_answers and 'q1' in previous_answers and not st.session_state.get(val_key):
            st.session_state[val_key] = previous_answers['q1']
            print(f"[DEBUG] Argument populated from previous session: {previous_answers['q1'][:50]}...")
        elif st.session_state.get(val_key):
            print(f"[DEBUG] Argument using existing session state value: {st.session_state.get(val_key)[:50]}...")
        
        # Use session state value
        current_value = st.session_state.get(val_key, '')
        print(f"[DEBUG] Current argument value: {current_value[:50]}..." if current_value else "[DEBUG] Current argument value: (empty)")
        answers['q1'] = st.text_area("Your Argument", value=current_value, key=key, height=200, 
                                      help="Present your position on this topic. Build a strong, logical argument.")
        st.session_state[val_key] = answers['q1']
        
        # Set q2 and q3 to empty for debate mode
        answers['q2'] = ''
        answers['q3'] = ''
        
        # After submission, show Devil's Advocate response (no score display)
        if fb and submitted:
            text = fb.get('new_feedback1', fb.get('feedback1', ''))
            if text:
                st.markdown(
                    f"""
                    <div style='background:rgba(255,90,90,0.15); color:#fff; border-radius:8px; padding:1.2rem; margin-bottom:0.7rem; margin-top:0.5rem; font-size:1.08rem; border-left: 4px solid rgba(255,90,90,0.6);'>
                        <b>😈 Devil's Advocate Response:</b><br><br>{text}
                    </div>
                    """, unsafe_allow_html=True
                )
                
                # Show buttons to continue debating or call the judge
                if not st.session_state.get('show_continue_debate', False) and not st.session_state.get('judge_mode', False):
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button('💬 Respond to Devil\'s Advocate', key='show_continue_btn', use_container_width=True, type='primary'):
                            st.session_state['show_continue_debate'] = True
                            # Scroll to continue section
                            if SCROLL_AVAILABLE:
                                st.session_state['scroll_to_continue'] = True
                            rerun()
                    with col2:
                        if st.button('⚖️ Call the Judge', key='call_judge_btn', use_container_width=True):
                            st.session_state['judge_mode'] = True
                            st.session_state['calling_judge'] = True
                            rerun()
            else:
                st.info("No response from Devil's Advocate yet")

        # --- Submission logic ---
        st.markdown("<div style='height:1.2rem;'></div>", unsafe_allow_html=True)
        fb = st.session_state.get('feedback')
        awaiting_resubmit = st.session_state.get('awaiting_resubmit', False)
        
        if not fb and not awaiting_resubmit:
            # Define callback function for submission
            def handle_submit():
                # Prevent submission if answer is empty
                if not answers['q1'].strip():
                    st.session_state['submit_error'] = 'Please provide your argument before submitting.'
                    return
                
                # Process submission
                with st.spinner('Submitting your answers...'):
                    user_arg = answers.get('q1', '')
                    
                    # FIRST: Add user's argument to context so Devil's Advocate can see it
                    if user_arg:
                        assignment_memory.add_student_answer('1', user_arg)
                        print(f"[DEBUG] Added user argument to context BEFORE calling Devil's Advocate: {user_arg[:50]}...")
                    
                    # Record answers to sheet
                    record_answers(exec_id, sid, aid, answers)
                    
                    # THEN: Call Devil's Advocate (now it can see the user's argument in history)
                    grade_res = run_grading(exec_id, sid, aid, answers)
                    
                    if grade_res:
                        # Queue grading data for background writing
                        background_writer.write_async('grading', grade_res)
                        
                        # Now add Devil's Advocate response to the context
                        devils_response = grade_res.get('feedback1', '')
                        if devils_response:
                            assignment_memory.add_grading_result('1', 0, devils_response)
                            print(f"[DEBUG] Added Devil's Advocate response to context: {devils_response[:50]}...")
                        
                        # Update started status to TRUE if it was FALSE
                        if started_status == 'FALSE':
                            try:
                                sheets.update_started_status(sid, aid, 'TRUE')
                                print(f"[SESSION] Updated started status to TRUE for student {sid}, assignment {aid}")
                            except Exception as e:
                                print(f"[ERROR] Failed to update started status: {e}")
                        
                        # Store in session state for the rest of the app
                        st.session_state['feedback'] = grade_res
                        st.session_state['submitted'] = True
                        st.session_state['awaiting_resubmit'] = False
                        st.session_state['submit_error'] = None
                    else:
                        st.session_state['submit_error'] = 'Failed to grade your answers. Please try again.'
            
            # Show Submit Answers button with callback
            submit = st.button('Submit Answers', use_container_width=True, on_click=handle_submit)
            
            # Show any submission errors
            if st.session_state.get('submit_error'):
                st.error(st.session_state['submit_error'])
        # --- Completion/Follow-up logic ---
        if fb:
            st.markdown("<div style='height:0.7rem;'></div>", unsafe_allow_html=True)
            # For debate mode, there's no score-based completion
            # The debate continues through conversation
            st.info("💬 Continue the debate by asking questions or making new arguments in the conversation section below.")
            # If awaiting_resubmit, show resubmit button above conversation
            if awaiting_resubmit:
                resubmit = st.button('Resubmit', key='resubmit_btn', use_container_width=True)
                if resubmit:
                    # Submit new answers
                    if not answers['q1'].strip():
                        st.toast('Please provide your argument before resubmitting.', icon='⚠️')
                    else:
                        with st.spinner('Submitting your answers...'):
                            user_arg = answers.get('q1', '')
                            
                            # Add user's argument to context first
                            if user_arg:
                                assignment_memory.add_student_answer('1', user_arg)
                                print(f"[DEBUG] Added user argument to context (resubmit)")
                            
                            record_answers(exec_id, sid, aid, answers)
                            grade_res = run_grading(exec_id, sid, aid, answers)
                            
                            if grade_res:
                                sheets.grading.append_row(grade_res)
                                
                                # Add Devil's Advocate response to context
                                devils_response = grade_res.get('feedback1', '')
                                if devils_response:
                                    assignment_memory.add_grading_result('1', 0, devils_response)
                                    print(f"[DEBUG] Added Devil's Advocate response to context (resubmit)")
                                
                                # Skip evaluation for faster response - use grading directly
                                st.session_state['feedback'] = grade_res
                                st.session_state['submitted'] = True
                                st.session_state['awaiting_resubmit'] = False
                            else:
                                st.error("Failed to generate Devil's Advocate response. Please try again.")
                        rerun()
                # Add enhanced feedback button
                col1, col2 = st.columns(2)
                with col1:
                    enhanced_feedback = st.button('Get Enhanced Feedback', key='enhanced_feedback_btn')
                    if enhanced_feedback:
                        with st.spinner('Getting enhanced feedback...'):
                            eval_res = run_evaluation(fb)
                            if eval_res:
                                # Queue evaluation data for background writing
                                background_writer.write_async('evaluation', eval_res)
                                st.session_state['feedback'] = eval_res
                                st.success('Enhanced feedback generated!')
                                rerun()
                            else:
                                st.error("Failed to get enhanced feedback.")
                
                with col2:
                    # Use a counter-based key to force clearing
                    conv_counter = st.session_state.get('conv_counter', 0)
                    user_q = st.text_input('Ask a follow-up question:', key=f'conv_{conv_counter}')
                
                # Only process if there's a new question and it hasn't been processed yet
                if user_q and user_q != st.session_state.get('last_processed_question', ''):
                    try:
                        conv_res = run_conversation(exec_id, sid, user_q)
                        if conv_res:
                            # Queue conversation data for background writing
                            background_writer.write_async('conversations', conv_res)
                            agent_msg = conv_res.get('agent_msg', conv_res.get('content', 'No response available'))
                            
                            # Add to conversation cache (both new and old systems)
                            assignment_memory.add_conversation(user_q, agent_msg)
                            context_cache.add_conversation(user_q, agent_msg)
                            
                            # Store the response in session state to persist it
                            st.session_state['last_conversation_response'] = agent_msg
                            
                            # Mark this question as processed and increment counter to clear input
                            st.session_state['last_processed_question'] = user_q
                            st.session_state['conv_counter'] = conv_counter + 1
                            
                            # Show success message and trigger rerun to clear input
                            st.success("Response generated!")
                            rerun()
                        else:
                            st.error("Failed to get a response. Please try again.")
                    except Exception as e:
                        st.error(f"Error during conversation: {e}")
                
                # Display the stored conversation response (from session state or previous data)
                conversation_response = st.session_state.get('last_conversation_response') or latest_conversation_response
                if conversation_response:
                    st.write("**AI Response:**")
                    st.write(conversation_response)
                # Show Retry button at the bottom if not awaiting_resubmit
                if not awaiting_resubmit:
                    retry = st.button('Retry', key='retry_btn', use_container_width=True)
                    if retry:
                        # Set retry mode to show new question blocks below
                        st.session_state['retry_mode'] = True
                        st.session_state['retry_counter'] = st.session_state.get('retry_counter', 0) + 1
                        
                        # Auto-scroll to retry section
                        if SCROLL_AVAILABLE:
                            st.session_state['scroll_to_retry'] = True
                        
                        rerun()

        # --- Continue Debate: Show response box after Devil's Advocate responds ---
        if fb and st.session_state.get('show_continue_debate', False):
            # Place scroll anchor at the beginning of continue debate section
            if SCROLL_AVAILABLE and st.session_state.get('scroll_to_continue', False):
                scroll_to_here(0, key='continue_section_anchor')
                st.session_state['scroll_to_continue'] = False
            
            st.markdown("<div style='height:2rem;'></div>", unsafe_allow_html=True)
            st.markdown("---")
            st.markdown(f"<h2 style='text-align: center; margin-bottom: 1.5rem;'>💬 Continue the Debate</h2>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: center; color: #666; margin-bottom: 2rem;'>Respond to the Devil's Advocate's counter-argument below.</p>", unsafe_allow_html=True)
            
            continue_answers: dict[str, str] = {}
            
            # Show debate topic and response box
            st.markdown(f"<div class='question-card' style='margin-bottom:0.8rem; font-size:1.15rem;'><b>Debate Topic:</b> {debate_topic}</div>", unsafe_allow_html=True)
            continue_key = f'continue_a1_counter{st.session_state.get("continue_counter", 0)}'
            continue_val_key = 'continue_q1_val'
            continue_answers['q1'] = st.text_area("Your Response to Devil's Advocate", 
                                                  value=st.session_state.get(continue_val_key, ''), 
                                                  key=continue_key, 
                                                  height=200,
                                                  help="Counter the Devil's Advocate's arguments and strengthen your position.")
            st.session_state[continue_val_key] = continue_answers['q1']
            
            # Set q2 and q3 to empty
            continue_answers['q2'] = ''
            continue_answers['q3'] = ''
            
            # Continue debate submission button
            col1, col2 = st.columns(2)
            with col1:
                continue_submit = st.button('Submit Response', key='continue_submit_btn', use_container_width=True, type='primary')
            with col2:
                collapse_continue = st.button('Collapse', key='collapse_continue_btn', use_container_width=True)
            
            if continue_submit:
                # Get the actual current value from session state
                user_response = continue_answers.get('q1', '').strip()
                
                # Prevent submission if answer is empty
                if not user_response:
                    st.toast('Please provide your response before submitting.', icon='⚠️')
                else:
                    # Build answers dict with current value
                    continue_submit_answers = {'q1': user_response, 'q2': '', 'q3': ''}
                    
                    # FIRST: Add user's response to context so Devil's Advocate can see it
                    if user_response:
                        assignment_memory.add_student_answer('1', user_response)
                        print(f"[DEBUG] Added user response to context BEFORE calling Devil's Advocate: {user_response[:50]}...")
                    
                    # Record new response to sheet
                    record_answers(exec_id, sid, aid, continue_submit_answers)
                    
                    # THEN: Get Devil's Advocate response
                    with st.spinner("Devil's Advocate is considering your response..."):
                        grade_res = run_grading(exec_id, sid, aid, continue_submit_answers)
                    
                    if grade_res:
                        # Queue grading data for background writing
                        background_writer.write_async('grading', grade_res)
                        
                        # Add Devil's Advocate's response to context
                        devils_response = grade_res.get('feedback1', '')
                        if devils_response:
                            assignment_memory.add_grading_result('1', 0, devils_response)
                            print(f"[DEBUG] Added Devil's Advocate response to context: {devils_response[:50]}...")
                        
                        # Update the top answer box with the new response
                        st.session_state['q1_val'] = continue_answers['q1']
                        print(f"[DEBATE] Updated argument with: {continue_answers['q1'][:50]}...")
                        
                        # Store new Devil's Advocate response
                        st.session_state['feedback'] = grade_res
                        st.session_state['submitted'] = True
                        st.session_state['show_continue_debate'] = False
                        
                        # Clear continue answer value
                        st.session_state['continue_q1_val'] = ''
                        
                        # Increment counter to force new keys on next continue
                        st.session_state['continue_counter'] = st.session_state.get('continue_counter', 0) + 1
                        print(f"[DEBATE] Incremented continue counter to {st.session_state['continue_counter']}")
                        
                        # Scroll to top to see updated argument and Devil's Advocate response
                        if SCROLL_AVAILABLE:
                            st.session_state['scroll_to_top'] = True
                        
                        st.success('Response submitted! Scroll up to see the Devil\'s Advocate\'s counter-argument.')
                        st.rerun()
                    else:
                        st.error("Failed to generate Devil's Advocate response. Please try again.")
            
            if collapse_continue:
                # Hide continue debate section (keep the text so it persists if reopened)
                st.session_state['show_continue_debate'] = False
                # Don't clear continue_q1_val - preserve user's in-progress typing
                rerun()

        # --- Judge Mode: Show judge ruling and handle follow-ups ---
        if st.session_state.get('judge_mode', False):
            st.markdown("<div style='height:2rem;'></div>", unsafe_allow_html=True)
            st.markdown("---")
            
            # Call judge if this is the first time or if we just got a response to judge's question
            if st.session_state.get('calling_judge', False):
                judge_result = run_judge(exec_id, sid, aid)
                st.session_state['judge_result'] = judge_result
                st.session_state['calling_judge'] = False
            
            judge_result = st.session_state.get('judge_result')
            
            if judge_result:
                # Display judge's ruling (use reasoning field if available, otherwise full response)
                st.markdown(f"<h2 style='text-align: center; margin-bottom: 1.5rem;'>⚖️ Judge's Ruling</h2>", unsafe_allow_html=True)
                
                # Use the parsed reasoning if available, otherwise show full response
                judge_response = judge_result.get('reasoning', judge_result.get('full_response', ''))
                
                # Clean up any JSON artifacts
                import re
                # Remove JSON structure if present
                judge_response = re.sub(r'```json\s*\{.*?\}\s*```', '', judge_response, flags=re.DOTALL)
                judge_response = re.sub(r'\{[^{}]*"decision_made"[^{}]*\}', '', judge_response, flags=re.DOTALL)
                judge_response = judge_response.strip()
                
                if not judge_response:
                    judge_response = judge_result.get('full_response', 'No ruling available')
                
                st.markdown(
                    f"""
                    <div style='background:rgba(255,215,0,0.15); color:#fff; border-radius:8px; padding:1.5rem; margin-bottom:1rem; font-size:1.08rem; border-left: 4px solid rgba(255,215,0,0.7);'>
                        <b>⚖️ Judge:</b><br><br>{judge_response}
                    </div>
                    """, unsafe_allow_html=True
                )
                
                # Check if decision was made
                if judge_result.get('decision_made'):
                    winner = judge_result.get('winner')
                    if winner == 'user':
                        st.success("🎉 Congratulations! The Judge has ruled in your favor!")
                    elif winner == 'devils_advocate':
                        st.info("😈 The Judge has ruled in favor of the Devil's Advocate.")
                    else:
                        st.warning("⚖️ The Judge could not determine a clear winner.")
                    
                    # End judge mode
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button('End Round', key='end_judge_btn', use_container_width=True, type='primary'):
                            st.session_state['judge_mode'] = False
                            st.session_state['judge_result'] = None
                            st.session_state['judge_history'] = ''
                            rerun()
                    with col2:
                        if st.button('Continue Debating', key='continue_after_judge_btn', use_container_width=True):
                            st.session_state['judge_mode'] = False
                            st.session_state['judge_result'] = None
                            rerun()
                
                # If judge has a follow-up question
                elif judge_result.get('follow_up_question'):
                    follow_up = judge_result.get('follow_up_question')
                    question_for = judge_result.get('question_for')
                    
                    st.markdown(f"<p style='text-align: center; color: #ffd700; margin-top: 1rem; font-size: 1.1rem;'><b>The Judge needs clarification...</b></p>", unsafe_allow_html=True)
                    
                    if question_for == 'user':
                        st.markdown(f"<p style='text-align: center; margin-bottom: 1.5rem;'>Please answer the Judge's question:</p>", unsafe_allow_html=True)
                        
                        judge_q_key = f'judge_user_response_{st.session_state.get("judge_counter", 0)}'
                        judge_val_key = 'judge_user_response_val'
                        user_response = st.text_area("Your Response to the Judge", 
                                                     value=st.session_state.get(judge_val_key, ''),
                                                     key=judge_q_key,
                                                     height=150,
                                                     help="Answer the Judge's question to strengthen your argument.")
                        st.session_state[judge_val_key] = user_response
                        
                        if st.button('Submit Response to Judge', key='submit_judge_user_btn', use_container_width=True, type='primary'):
                            if not user_response.strip():
                                st.toast('Please provide a response to the Judge.', icon='⚠️')
                            else:
                                # Add to judge history
                                judge_hist = st.session_state.get('judge_history', '')
                                judge_hist += f"\n\nJudge asked User: {follow_up}\nUser responded: {user_response}"
                                st.session_state['judge_history'] = judge_hist
                                
                                # Clear response value and call judge again
                                st.session_state['judge_user_response_val'] = ''
                                st.session_state['judge_counter'] = st.session_state.get('judge_counter', 0) + 1
                                st.session_state['calling_judge'] = True
                                rerun()
                    
                    elif question_for == 'devils_advocate':
                        st.markdown(f"<p style='text-align: center; margin-bottom: 1.5rem;'>The Devil's Advocate must respond...</p>", unsafe_allow_html=True)
                        
                        # Show spinner and get Devil's Advocate response automatically
                        with st.spinner("😈 Devil's Advocate is responding to the Judge..."):
                            # Create a context for Devil's Advocate to answer the judge's question
                            da_answer_context = {
                                'q1': f"Judge's question: {follow_up}\n\nPlease provide a clear, direct answer that supports your position in the debate."
                            }
                            
                            # Use the grading function but with different context
                            da_response = run_grading(exec_id, sid, aid, da_answer_context)
                            da_answer = da_response.get('feedback1', 'No response generated')
                            
                            # Display Devil's Advocate's response
                            st.markdown(
                                f"""
                                <div style='background:rgba(255,90,90,0.15); color:#fff; border-radius:8px; padding:1.2rem; margin-bottom:0.7rem; margin-top:0.5rem; font-size:1.08rem; border-left: 4px solid rgba(255,90,90,0.6);'>
                                    <b>😈 Devil's Advocate Response:</b><br><br>{da_answer}
                                </div>
                                """, unsafe_allow_html=True
                            )
                            
                            # Add to judge history
                            judge_hist = st.session_state.get('judge_history', '')
                            judge_hist += f"\n\nJudge asked Devil's Advocate: {follow_up}\nDevil's Advocate responded: {da_answer}"
                            st.session_state['judge_history'] = judge_hist
                            
                            # Show button to have judge review the response
                            if st.button('Judge Reviews Response', key='judge_review_da_btn', use_container_width=True, type='primary'):
                                st.session_state['calling_judge'] = True
                                rerun()

    # --- Footer ---
    st.markdown("""
    <div class='footer'>
        <span>&middot; <a href='https://github.com/mudcario350/streamlitapp' target='_blank'>GitHub</a></span>
    </div>
    """, unsafe_allow_html=True)

# Launch with guaranteed write completion
def run_app():
    """Run the app with proper cleanup."""
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 App interrupted by user")
    except Exception as e:
        print(f"❌ App error: {e}")
    finally:
        print("🔄 Ensuring all writes complete before shutdown...")
        try:
            background_writer.shutdown()
            print("✅ All writes completed successfully")
        except Exception as e:
            print(f"⚠️ Error during shutdown: {e}")

# Run the app
run_app()

