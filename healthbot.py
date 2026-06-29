"""
HealthBot: AI-Powered Patient Education System
MediTech Solutions - LangGraph Workflow Implementation
"""

# ============================================================
# CELL 1: Load Environment Variables
# ============================================================

from dotenv import load_dotenv
import os

# Load OpenAI and Tavily API keys.
load_dotenv('config.env')

assert os.getenv('OPENAI_API_KEY') is not None, "OPENAI_API_KEY not found in config.env"
assert os.getenv('TAVILY_API_KEY') is not None, "TAVILY_API_KEY not found in config.env"

print("✅ API keys loaded successfully.")


# ============================================================
# CELL 2: Imports
# ============================================================

from typing import TypedDict, Optional, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage


# ============================================================
# CELL 3: Define HealthBot State
# ============================================================

class HealthBotState(TypedDict):
    """
    State object shared across all LangGraph nodes.
    Each field is updated by the appropriate node and
    read by subsequent nodes.
    """
    # Conversation messages (tool calls, AI responses, etc.)
    messages: Annotated[list, add_messages]

    # The health topic the patient wants to learn about
    topic: Optional[str]

    # Raw Tavily search results
    search_results: Optional[str]

    # Patient-friendly summarization (3-4 paragraphs)
    summary: Optional[str]

    # Generated quiz question
    quiz_question: Optional[str]

    # Patient's answer to the quiz
    patient_answer: Optional[str]

    # Grade + justification from the model
    grade_and_feedback: Optional[str]

    # Flow control: "new_topic" or "exit"
    next_action: Optional[str]


# ============================================================
# CELL 4: Initialize Model and Tavily Tool
# ============================================================

# Initialize the OpenAI chat model
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3,
    api_key=os.getenv("OPENAI_API_KEY"),
)

# Initialize Tavily search tool (focuses on reputable medical sources)
tavily_tool = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_domains=[
        "mayoclinic.org",
        "webmd.com",
        "medlineplus.gov",
        "healthline.com",
        "nih.gov",
        "cdc.gov",
        "who.int",
        "medicalnewstoday.com",
        "clevelandclinic.org",
        "hopkinsmedicine.org"
    ],
    tavily_api_key=os.getenv("TAVILY_API_KEY")
)

print("✅ OpenAI model and Tavily tool initialized.")


# ============================================================
# CELL 5: Define LangGraph Nodes
# ============================================================

def node_collect_topic(state: HealthBotState) -> HealthBotState:
    """
    Node 1: Ask the patient what health topic they'd like to learn about.
    Uses Jupyter's input() for interactive modal input.
    """
    print("\n" + "="*60)
    print("🏥  Welcome to HealthBot — Your Personal Health Educator")
    print("="*60)
    print("\nI'm here to help you understand medical topics in plain,")
    print("easy-to-understand language. Let's get started!\n")

    topic = input("📋 What health topic or medical condition would you like to learn about today? ")

    print(f"\n🔍 Great! Looking up information on: '{topic}' ...")

    return {
        **state,
        "topic": topic,
        "messages": [HumanMessage(content=f"I want to learn about: {topic}")]
    }


def node_search_tavily(state: HealthBotState) -> HealthBotState:
    """
    Node 2: Use Tavily to search for relevant, up-to-date medical information.
    The LLM calls Tavily as a tool to retrieve reputable medical content.
    """
    topic = state["topic"]

    # Bind Tavily tool to LLM so it can call it
    llm_with_tools = llm.bind_tools([tavily_tool])

    search_prompt = [
        SystemMessage(content=(
            "You are a medical research assistant. Your job is to search for "
            "accurate, up-to-date medical information from reputable health sources. "
            "When given a health topic, use the tavily_search_results_json tool to "
            "find comprehensive information about it. Focus on reputable sources like "
            "Mayo Clinic, WebMD, NIH, CDC, WHO, and similar authoritative health sites."
        )),
        HumanMessage(content=(
            f"Please search for detailed medical information about: {topic}. "
            "Find information about what it is, causes, symptoms, treatments, "
            "and prevention if applicable."
        ))
    ]

    # LLM decides to call Tavily
    response = llm_with_tools.invoke(search_prompt)

    # Extract tool calls and execute them
    search_results_text = ""

    if response.tool_calls:
        for tool_call in response.tool_calls:
            if tool_call["name"] == "tavily_search_results_json":
                raw_results = tavily_tool.invoke(tool_call["args"])
                # Format results into readable text
                for i, result in enumerate(raw_results, 1):
                    search_results_text += f"\n[Source {i}]: {result.get('url', 'Unknown')}\n"
                    search_results_text += f"{result.get('content', '')}\n"
                    search_results_text += "-" * 40 + "\n"

    print("✅ Medical information retrieved from reputable sources.")

    return {
        **state,
        "search_results": search_results_text,
        "messages": state["messages"] + [
            response,
            ToolMessage(
                content=search_results_text,
                tool_call_id=response.tool_calls[0]["id"] if response.tool_calls else "search_001"
            )
        ]
    }


def node_summarize(state: HealthBotState) -> HealthBotState:
    """
    Node 3: Summarize the Tavily search results into a patient-friendly explanation.
    The model uses ONLY the search results — no outside knowledge.
    """
    topic = state["topic"]
    search_results = state["search_results"]

    summary_prompt = [
        SystemMessage(content=(
            "You are a compassionate patient educator working at a hospital. "
            "Your job is to take raw medical search results and rewrite them into "
            "a clear, warm, patient-friendly summary. Important rules:\n"
            "1. Use ONLY the information provided in the search results — do NOT add outside knowledge.\n"
            "2. Write in simple, plain English — avoid jargon; explain any medical terms used.\n"
            "3. Write exactly 3 to 4 paragraphs.\n"
            "4. Be empathetic, clear, and encouraging in tone.\n"
            "5. Structure: (a) What it is, (b) Causes/Risk Factors, (c) Symptoms/Diagnosis, (d) Treatment/Prevention.\n"
            "6. End with a brief reminder to always consult a healthcare professional."
        )),
        HumanMessage(content=(
            f"Here are the search results about '{topic}':\n\n"
            f"{search_results}\n\n"
            "Please summarize this into a patient-friendly explanation of 3-4 paragraphs "
            "using ONLY the information above."
        ))
    ]

    response = llm.invoke(summary_prompt)
    summary = response.content

    return {
        **state,
        "summary": summary,
        "messages": state["messages"] + [response]
    }


def node_present_summary(state: HealthBotState) -> HealthBotState:
    """
    Node 4 & 5: Present the summarized information to the patient,
    then prompt them to indicate they're ready for the comprehension check.
    """
    topic = state["topic"]
    summary = state["summary"]

    print("\n" + "="*60)
    print(f"📖  Health Information: {topic.title()}")
    print("="*60)
    print(f"\n{summary}\n")
    print("="*60)
    print("⚠️  Note: This information is for educational purposes only.")
    print("    Always consult your healthcare provider for medical advice.")
    print("="*60)

    # Prompt 5: Wait for patient to be ready
    input("\n✅ Press ENTER when you've finished reading and are ready for a quick comprehension check...")

    print("\n📝 Great! Let me generate a question based on what you just read...")

    return state


def node_generate_quiz(state: HealthBotState) -> HealthBotState:
    """
    Node 6: Generate a single, relevant quiz question based ONLY on the summary.
    """
    summary = state["summary"]
    topic = state["topic"]

    quiz_prompt = [
        SystemMessage(content=(
            "You are a medical educator creating a comprehension quiz for a patient. "
            "Your task is to write ONE clear, fair quiz question. Rules:\n"
            "1. Base the question ONLY on information in the provided summary — nothing else.\n"
            "2. The question must be directly answerable from the summary alone.\n"
            "3. Ask about an important concept (not trivial detail).\n"
            "4. Write an open-ended question (not multiple choice) so the patient can answer in their own words.\n"
            "5. Output ONLY the question — no preamble, no answer, no explanation."
        )),
        HumanMessage(content=(
            f"Based on this patient education summary about '{topic}':\n\n"
            f"{summary}\n\n"
            "Write ONE comprehension quiz question."
        ))
    ]

    response = llm.invoke(quiz_prompt)
    quiz_question = response.content.strip()

    return {
        **state,
        "quiz_question": quiz_question,
        "messages": state["messages"] + [response]
    }


def node_present_quiz(state: HealthBotState) -> HealthBotState:
    """
    Node 7 & 8: Present the quiz question and collect the patient's answer.
    """
    quiz_question = state["quiz_question"]

    print("\n" + "="*60)
    print("🧠  Comprehension Check")
    print("="*60)
    print(f"\n❓ {quiz_question}\n")

    patient_answer = input("✏️  Your answer: ")

    print("\n⏳ Evaluating your response...")

    return {
        **state,
        "patient_answer": patient_answer,
        "messages": state["messages"] + [HumanMessage(content=patient_answer)]
    }


def node_grade_answer(state: HealthBotState) -> HealthBotState:
    """
    Node 9: Evaluate the patient's response using ONLY the summary.
    Provides a letter grade (A–F) and justification with citations from the summary.
    """
    summary = state["summary"]
    quiz_question = state["quiz_question"]
    patient_answer = state["patient_answer"]

    grading_prompt = [
        SystemMessage(content=(
            "You are a compassionate medical educator grading a patient's quiz answer. "
            "Rules:\n"
            "1. Grade using ONLY the provided health information summary as your answer key.\n"
            "2. Assign a letter grade: A (excellent), B (good), C (partial), D (minimal), F (incorrect/missing).\n"
            "3. Provide a warm, encouraging explanation of the grade.\n"
            "4. Include specific citations/quotes from the summary to show what the correct answer is.\n"
            "5. If the answer is partially correct, acknowledge what was right before explaining what was missing.\n"
            "6. Format your response as:\n"
            "   GRADE: [letter]\n"
            "   FEEDBACK: [your explanation with citations]\n"
            "   CORRECT ANSWER: [brief correct answer based on the summary]"
        )),
        HumanMessage(content=(
            f"Health Information Summary:\n{summary}\n\n"
            f"Quiz Question: {quiz_question}\n\n"
            f"Patient's Answer: {patient_answer}\n\n"
            "Please grade this answer and provide feedback with citations from the summary."
        ))
    ]

    response = llm.invoke(grading_prompt)
    grade_and_feedback = response.content

    return {
        **state,
        "grade_and_feedback": grade_and_feedback,
        "messages": state["messages"] + [response]
    }


def node_present_grade(state: HealthBotState) -> HealthBotState:
    """
    Node 10 & 11: Present the grade/feedback, then ask if the patient
    wants to learn about another topic or exit.
    """
    grade_and_feedback = state["grade_and_feedback"]

    print("\n" + "="*60)
    print("📊  Your Results")
    print("="*60)
    print(f"\n{grade_and_feedback}\n")
    print("="*60)
    print("\n💙 Great effort! Remember, learning about your health is the")
    print("   first step toward better wellbeing.")

    print("\n" + "-"*60)
    print("What would you like to do next?")
    print("  1. Learn about a new health topic")
    print("  2. Exit HealthBot")
    print("-"*60)

    while True:
        choice = input("\n👉 Enter 1 or 2: ").strip()
        if choice == "1":
            next_action = "new_topic"
            break
        elif choice == "2":
            next_action = "exit"
            break
        else:
            print("   Please enter 1 or 2.")

    return {
        **state,
        "next_action": next_action
    }


def node_exit(state: HealthBotState) -> HealthBotState:
    """
    Node 12b: End the session gracefully.
    """
    print("\n" + "="*60)
    print("👋  Thank you for using HealthBot!")
    print("="*60)
    print("\nStay informed, stay healthy. Remember to always consult")
    print("your healthcare provider for personalized medical advice.\n")
    print("Goodbye! 💙\n")
    return state


def node_reset_state(state: HealthBotState) -> HealthBotState:
    """
    Node 12a: Reset state completely when starting a new topic.
    Clears all previous health information to maintain patient privacy
    and accuracy — no data from the previous session carries over.
    """
    print("\n" + "="*60)
    print("🔄  Starting a new session...")
    print("="*60)
    print("✅ Previous session data cleared for your privacy.\n")

    # Return a completely fresh state
    return {
        "messages": [],
        "topic": None,
        "search_results": None,
        "summary": None,
        "quiz_question": None,
        "patient_answer": None,
        "grade_and_feedback": None,
        "next_action": None
    }


# ============================================================
# CELL 6: Define Routing Logic (Conditional Edges)
# ============================================================

def route_after_grade(state: HealthBotState) -> str:
    """
    Conditional edge: After presenting the grade, route to either
    'reset' (new topic) or 'exit' based on patient choice.
    """
    if state.get("next_action") == "new_topic":
        return "reset_state"
    else:
        return "exit"


# ============================================================
# CELL 7: Build the LangGraph Workflow
# ============================================================

def build_healthbot_graph() -> StateGraph:
    """
    Construct and compile the LangGraph StateGraph with all nodes and edges.
    """
    graph = StateGraph(HealthBotState)

    # --- Add all nodes ---
    graph.add_node("collect_topic",    node_collect_topic)
    graph.add_node("search_tavily",    node_search_tavily)
    graph.add_node("summarize",        node_summarize)
    graph.add_node("present_summary",  node_present_summary)
    graph.add_node("generate_quiz",    node_generate_quiz)
    graph.add_node("present_quiz",     node_present_quiz)
    graph.add_node("grade_answer",     node_grade_answer)
    graph.add_node("present_grade",    node_present_grade)
    graph.add_node("reset_state",      node_reset_state)
    graph.add_node("exit",             node_exit)

    # --- Set entry point ---
    graph.set_entry_point("collect_topic")

    # --- Sequential edges (happy path) ---
    graph.add_edge("collect_topic",   "search_tavily")
    graph.add_edge("search_tavily",   "summarize")
    graph.add_edge("summarize",       "present_summary")
    graph.add_edge("present_summary", "generate_quiz")
    graph.add_edge("generate_quiz",   "present_quiz")
    graph.add_edge("present_quiz",    "grade_answer")
    graph.add_edge("grade_answer",    "present_grade")

    # --- Conditional edge: new topic OR exit ---
    graph.add_conditional_edges(
        "present_grade",
        route_after_grade,
        {
            "reset_state": "reset_state",
            "exit":        "exit"
        }
    )

    # --- After reset, loop back to collect new topic ---
    graph.add_edge("reset_state", "collect_topic")

    # --- Exit leads to END ---
    graph.add_edge("exit", END)

    return graph.compile()


# Build the graph
healthbot = build_healthbot_graph()

print("✅ HealthBot LangGraph workflow compiled successfully.")
print("\nGraph nodes:", [
    "collect_topic", "search_tavily", "summarize",
    "present_summary", "generate_quiz", "present_quiz",
    "grade_answer", "present_grade", "reset_state", "exit"
])


# ============================================================
# CELL 8: Run the HealthBot
# ============================================================

def run_healthbot():
    """
    Initialize and run the HealthBot workflow with a fresh state.
    """
    # Fresh initial state
    initial_state: HealthBotState = {
        "messages": [],
        "topic": None,
        "search_results": None,
        "summary": None,
        "quiz_question": None,
        "patient_answer": None,
        "grade_and_feedback": None,
        "next_action": None
    }

    print("\n" + "🏥 " * 20)
    print("    Starting HealthBot Session...")
    print("🏥 " * 20 + "\n")

    # Execute the graph — it runs until END
    final_state = healthbot.invoke(initial_state)

    return final_state


if __name__ == "__main__":
    run_healthbot()
