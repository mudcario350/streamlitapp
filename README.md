# Dynamic Assignment (Devil's Advocate) App

A sophisticated AI-powered debate simulation system where the AI plays Devil's Advocate against student arguments. Built with Streamlit, LangGraph, and Google Sheets integration.

## Features

- **Devil's Advocate Debate System**: AI argues against student positions
- **Evidence-Based Judging**: Judge evaluates arguments based on concrete evidence
- **3-Claim Victory Rule**: First to present 3 well-supported claims wins
- **Interactive Conversations**: AI guides debate with follow-up questions
- **Memory Management**: LangGraph-based memory system tracks debate flow
- **Google Sheets Integration**: All data stored and managed in Google Sheets

## Architecture

### Tech Stack
- **Frontend**: Streamlit
- **AI Orchestration**: LangGraph (native, not n8n)
- **LLM Providers**: OpenAI, Google Gemini
- **Data Storage**: Google Sheets (via gspread)
- **Memory**: LangGraph MemorySaver

### Key Components
- `app.py` - Compiled Streamlit application
- `da_config.py` - Configuration for DA app
- `da_prompt_manager.py` - Devil's Advocate prompt management
- `da_source_app.py` - Source code for DA app
- `deploy.sh` - Deployment script with DA mode

## Setup

See setup documentation in:
- `GEMINI_SETUP.md` - Google Gemini configuration
- `GOOGLE_DOCS_SETUP.md` - Google Docs integration
- `MEMORY_IMPLEMENTATION_SUMMARY.md` - LangGraph memory system
- `PARALLEL_INTEGRATION_GUIDE.md` - Parallel processing architecture

## Deployment

```bash
./deploy.sh -m da "Your commit message"
```

This compiles `da_config.py`, `da_prompt_manager.py`, and `da_source_app.py` into `app.py` and deploys.

## Related Projects

- **Quiz App** (`../quiz_app/`): Variable question quiz/assignment system with 1-25 questions support

## License

Proprietary - All rights reserved
