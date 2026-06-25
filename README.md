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
Edit `config.env` with your actual keys:
```
OPENAI_API_KEY="sk-your-actual-openai-key"
TAVILY_API_KEY="tvly-your-actual-tavily-key"
```
- Get OpenAI key: https://platform.openai.com/api-keys
- Get Tavily key: https://app.tavily.com/home (first 1000 requests free)

### 5. Launch the notebook
```bash
jupyter notebook HealthBot.ipynb
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

## State Object

| Field               | Type          | Set By            | Used By                        |
|---------------------|---------------|-------------------|-------------------------------|
| `messages`          | list          | All nodes         | All nodes (full history)       |
| `topic`             | str           | collect_topic     | search_tavily, summarize       |
| `search_results`    | str           | search_tavily     | summarize                      |
| `summary`           | str           | summarize         | present_summary, generate_quiz, grade_answer |
| `quiz_question`     | str           | generate_quiz     | present_quiz, grade_answer     |
| `patient_answer`    | str           | present_quiz      | grade_answer                   |
| `grade_and_feedback`| str           | grade_answer      | present_grade                  |
| `next_action`       | str           | present_grade     | route_after_grade (router)     |

---

## Node Responsibilities

| Node | Responsibility |
|------|---------------|
| `collect_topic` | Patient input: health topic |
| `search_tavily` | LLM tool call → Tavily → raw medical data |
| `summarize` | LLM → 3-4 paragraph patient summary |
| `present_summary` | Display summary + wait for patient readiness |
| `generate_quiz` | LLM → 1 open-ended question from summary |
| `present_quiz` | Display question + collect answer |
| `grade_answer` | LLM → A-F grade + cited feedback |
| `present_grade` | Display grade + route decision |
| `reset_state` | Clear ALL state (privacy + accuracy) |
| `exit` | Goodbye message → END |

---

## Key Design Decisions

- **Single-responsibility nodes** — each node does exactly one thing
- **State reset on new topic** — `node_reset_state` returns a completely fresh `HealthBotState` so no data from the previous session bleeds into the next
- **Reputable sources only** — Tavily's `include_domains` restricts to Mayo Clinic, NIH, CDC, WHO, WebMD, and 5 other authoritative medical sources
- **Summary-only grading** — both quiz generation and grading prompts explicitly instruct the LLM to use ONLY the summary, not outside knowledge
- **Cited feedback** — grading node prompts the LLM to quote specific lines from the summary in its justification
