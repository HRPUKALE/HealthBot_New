# 🏥 HealthBot: AI-Powered Patient Education System
**MediTech Solutions — LangGraph Workflow Prototype**

---

## Overview

HealthBot is a LangGraph-based conversational workflow that provides personalized, on-demand health information to patients. It searches reputable medical sources, summarizes results in patient-friendly language, administers comprehension quizzes, and provides graded feedback with citations.

---

## Setup Instructions

### 1. Initialize uv and virtual environment
```bash
pip install uv
uv init
uv venv --python 3.11.13
```

### 2. Activate the virtual environment
**Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate
```
**Mac/Linux:**
```bash
source .venv/bin/activate
```

### 3. Install dependencies
```bash
uv add -r requirements.txt
```

### 4. Configure API keys
Copy the example config and add your keys:
```bash
cp config.env.example config.env
```
Edit `config.env`:
```
OPENAI_API_KEY="sk-your-actual-openai-key"
TAVILY_API_KEY="tvly-your-actual-tavily-key"
```
- Get OpenAI key: https://platform.openai.com/api-keys
- Get Tavily key: https://app.tavily.com/home (first 1000 requests free)

### 5. Run HealthBot
```bash
python healthbot.py
```

---

## LangGraph Workflow

```
[START]
   ↓
collect_topic      ← Asks patient for health topic (input())
   ↓
search_tavily      ← LLM calls Tavily tool; retrieves from 10 medical domains
   ↓
summarize          ← LLM writes 3-4 paragraph patient-friendly summary
   ↓
present_summary    ← Prints summary; waits for patient ENTER (input())
   ↓
generate_quiz      ← LLM creates 1 open-ended question from summary only
   ↓
present_quiz       ← Prints question; collects patient answer (input())
   ↓
grade_answer       ← LLM grades A-F with justification + citations
   ↓
present_grade      ← Prints grade; asks: 1=new topic, 2=exit (input())
   ↙               ↘
reset_state        exit
   ↓                ↓
collect_topic     [END]
```

---

## Key Design Decisions

- **Single-responsibility nodes** — each node does exactly one thing
- **State reset on new topic** — clears all state for privacy between sessions
- **Reputable sources only** — Tavily restricted to Mayo Clinic, NIH, CDC, WHO, WebMD, and 5 other medical sources
- **Summary-only grading** — quiz and grading use only retrieved summary content
- **Cited feedback** — grades include quotes from the summary
