#!/usr/bin/env python3
"""
app.py – Interactive CLI for the session-scoped astrology RAG chatbot.

Flow:
  1. Prompt for name, DOB, birth time, birth place.
  2. Run the setup phase (resolve rows, build session vector store).
  3. Loop: ask questions -> retrieve -> answer.
  4. Type 'quit' to exit -> cleanup.

Usage:
    python app.py
"""

from __future__ import annotations

import logging
import sys

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    # Late import so .env is loaded first
    from standard.workflow import build_setup_graph_standard as setup_graph

    print("\n" + "=" * 60)
    print("  Vedic Astrology RAG Chatbot")
    print("  Session-Scoped Retrieval")
    print("=" * 60)

    # ── Step 1: Collect birth details ─────────────────────────────────
    print("\nPlease provide your birth details:\n")

    name = input("  Name: ").strip()
    if not name:
        print("  Name is required. Exiting.")
        sys.exit(1)

    dob = input("  Date of Birth (YYYY-MM-DD): ").strip()
    if not dob:
        print("  DOB is required. Exiting.")
        sys.exit(1)

    birth_time = input("  Birth Time (HH:MM, 24h) [12:00]: ").strip() or "12:00"
    birth_place = input("  Birth Place [Unknown]: ").strip() or "Unknown"

    # ── Step 2: Run setup phase ───────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Setting up your natal chart session...")
    print(f"{'='*60}\n")

    initial_state = {
        "session_id": "",
        "birth_details": {
            "name": name,
            "dob": dob,
            "birth_time": birth_time,
            "birth_place": birth_place,
        },
        "selected_row_refs": [],
        "session_documents": [],
        "messages": [],
        "retrieved_docs": [],
        "final_answer": "",
    }

    try:
        state = setup_graph.invoke(initial_state)
    except Exception as e:
        print(f"\n  Error during setup: {e}")
        sys.exit(1)

    session_id = state["session_id"]
    num_docs = len(state["session_documents"])
    num_refs = len(state["selected_row_refs"])

    print(f"\n{'='*60}")
    print(f"  Session ready!")
    print(f"  Session ID:  {session_id}")
    print(f"  Row refs:    {num_refs}")
    print(f"  Documents:   {num_docs}")
    print(f"{'='*60}")

    if num_docs == 0:
        print("\n  Warning: No documents were loaded for this session.")
        print("  Answers may use the fallback message.\n")

    # ── Step 3: Question loop ─────────────────────────────────────────
    print("\n  Ask me anything about your chart!")
    print("  Type 'quit' or 'exit' to end the session.\n")

    chat_history = []

    while True:
        try:
            question = input("  You: ").strip()
        except (KeyboardInterrupt, EOFError):
            question = "quit"

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            break

        chat_history.append({"role": "user", "content": question})

        query_state = {
            "session_id": session_id,
            "birth_details": state["birth_details"],
            "selected_row_refs": state["selected_row_refs"],
            "session_documents": state["session_documents"],
            "messages": chat_history,
            "retrieved_docs": [],
            "final_answer": "",
        }

        try:
            result = query_graph.invoke(query_state)
            answer = result["final_answer"]
        except Exception as e:
            answer = f"Sorry, I encountered an error: {e}"
            logger.error("Query error: %s", e)

        chat_history.append({"role": "assistant", "content": answer})
        k_used = len(result.get("retrieved_docs", []))

        print(f"\n  Astrologer ({k_used} refs): {answer}\n")

    # ── Step 4: Cleanup ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Ending session...")

    cleanup_graph.invoke({
        "session_id": session_id,
        "birth_details": {},
        "selected_row_refs": [],
        "session_documents": [],
        "messages": [],
        "retrieved_docs": [],
        "final_answer": "",
    })

    print(f"  Session '{session_id}' cleaned up. Goodbye!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
