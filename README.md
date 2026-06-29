# HealthBot: AI-Powered Patient Education System

LangGraph workflow for patient education: Tavily search, patient-friendly summaries, comprehension quizzes, and graded feedback.

## Setup

```powershell
pip install uv
uv init
uv venv --python 3.11.13
.\.venv\Scripts\Activate
uv add -r requirements.txt
```

Copy `config.env.example` to `config.env` and add your API keys:

```
OPENAI_API_KEY="sk-your-key"
TAVILY_API_KEY="tvly-your-key"
```

## Run

```powershell
jupyter notebook healthbot.ipynb
```

Run all cells and follow the `input()` prompts.
