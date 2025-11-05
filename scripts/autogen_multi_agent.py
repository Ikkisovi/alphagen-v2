"""Multi-agent chat harness built on Microsoft Autogen.

This script wires up three specialized agents for collaborative problem solving:

* GeminiReader – uses Google Gemini for long-form document digestion and organization.
* ClaudePlanner – uses Anthropic Claude for planning and coding work.
* CodexAdvisor – uses OpenAI GPT models (Codex-inspired) to critique and suggest improvements.

Usage:
    python scripts/autogen_multi_agent.py --task "<prompt>" [--doc path/to/context.txt]

Environment variables:
    OPENAI_API_KEY     – OpenAI API key for the CodexAdvisor and chat manager.
    ANTHROPIC_API_KEY  – Anthropic API key for the ClaudePlanner.
    GOOGLE_API_KEY     – Google Generative AI key for the GeminiReader.

You can copy autogen.env.example to .env (or pass --env-file) and fill in your keys.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv

from autogen import (
    AssistantAgent,
    GroupChat,
    GroupChatManager,
    LLMConfig,
    UserProxyAgent,
)
from autogen.oai import (
    AnthropicLLMConfigEntry,
    GeminiLLMConfigEntry,
    OpenAILLMConfigEntry,
)


DEFAULT_GEMINI_MODEL = "gemini-1.5-pro"
DEFAULT_CLAUDE_MODEL = "claude-3-5-sonnet-20241022"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a multi-agent Autogen chat with Gemini, Claude, and Codex roles."
    )
    parser.add_argument(
        "--task",
        required=True,
        help="High-level instruction for the team. Use quotes for multi-line prompts.",
    )
    parser.add_argument(
        "--doc",
        type=Path,
        help="Optional path to a supporting document that Gemini should digest first.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Optional .env file to load in addition to environment variables (default: ./.env).",
    )
    parser.add_argument(
        "--gemini-model",
        default=DEFAULT_GEMINI_MODEL,
        help=f"Gemini model for GeminiReader (default: {DEFAULT_GEMINI_MODEL}).",
    )
    parser.add_argument(
        "--claude-model",
        default=DEFAULT_CLAUDE_MODEL,
        help=f"Claude model for ClaudePlanner (default: {DEFAULT_CLAUDE_MODEL}).",
    )
    parser.add_argument(
        "--openai-model",
        default=DEFAULT_OPENAI_MODEL,
        help=f"OpenAI model for CodexAdvisor and manager (default: {DEFAULT_OPENAI_MODEL}).",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=12,
        help="Maximum number of conversation rounds before forcing termination.",
    )
    return parser.parse_args()


def ensure_env(var_name: str) -> str:
    """Fetch an environment variable or raise a helpful error."""
    value = os.getenv(var_name)
    if not value:
        raise RuntimeError(
            f"Missing environment variable {var_name}. "
            "Populate it in your environment or .env file."
        )
    return value


def read_document(path: Path | None) -> str:
    if path is None:
        return ""
    if not path.exists():
        raise FileNotFoundError(f"Document path not found: {path}")
    # We only expect UTF-8 text assets for now.
    return path.read_text(encoding="utf-8")


def build_agents(args: argparse.Namespace) -> tuple[UserProxyAgent, list[AssistantAgent], GroupChatManager]:
    """Instantiate the user proxy, specialized assistants, and the group chat manager."""
    gemini_llm = LLMConfig(
        GeminiLLMConfigEntry(
            model=args.gemini_model,
            api_key=ensure_env("GOOGLE_API_KEY"),
            stream=False,
        ),
        temperature=0.4,
    )

    claude_llm = LLMConfig(
        AnthropicLLMConfigEntry(
            model=args.claude_model,
            api_key=ensure_env("ANTHROPIC_API_KEY"),
        ),
        temperature=0.2,
    )

    openai_llm = LLMConfig(
        OpenAILLMConfigEntry(
            model=args.openai_model,
            api_key=ensure_env("OPENAI_API_KEY"),
        ),
        temperature=0.5,
    )

    gemini_reader = AssistantAgent(
        name="GeminiReader",
        system_message=(
            "Role: Long-form reader & knowledge organizer.\n"
            "You ingest provided context, extract the key structure, and deliver a concise outline.\n"
            "Surface the highest-signal passages and unresolved questions. Keep outputs structured."
        ),
        llm_config=gemini_llm,
        human_input_mode="NEVER",
    )

    claude_planner = AssistantAgent(
        name="ClaudePlanner",
        system_message=(
            "Role: Planner & coder.\n"
            "Synthesize GeminiReader's findings into an actionable plan before writing code snippets.\n"
            "When coding, provide well-commented solutions and call out assumptions and TODOs explicitly."
        ),
        llm_config=claude_llm,
        human_input_mode="NEVER",
    )

    codex_advisor = AssistantAgent(
        name="CodexAdvisor",
        system_message=(
            "Role: Critical advisor.\n"
            "Stress-test the plan and code, point out risks, and suggest refinements. "
            "Be direct but constructive. Summarize final recommendations before calling TERMINATE."
        ),
        llm_config=openai_llm,
        human_input_mode="NEVER",
    )

    def is_termination_msg(msg: dict[str, str | None]) -> bool:
        content = (msg.get("content") or "").strip().upper()
        return content.endswith("TERMINATE")

    user_proxy = UserProxyAgent(
        name="UserProxy",
        system_message=(
            "You represent the end user. Forward the overall task and supporting materials, "
            "then stay silent. The agents must end by replying with TERMINATE once done."
        ),
        human_input_mode="NEVER",
        is_termination_msg=is_termination_msg,
    )

    group_chat = GroupChat(
        agents=[gemini_reader, claude_planner, codex_advisor],
        max_round=args.max_rounds,
        speaker_selection_method="round_robin",
    )

    manager = GroupChatManager(
        groupchat=group_chat,
        llm_config=openai_llm,
        system_message=(
            "You orchestrate the collaboration. Ensure each agent respects its role and "
            "the discussion converges to a clear solution. Terminate promptly when objectives are met."
        ),
        human_input_mode="NEVER",
    )

    return user_proxy, [gemini_reader, claude_planner, codex_advisor], manager


def build_initial_message(task: str, doc_content: str) -> str:
    if not doc_content:
        return task
    header = "Supporting document:\n" + "-" * 80
    footer = "-" * 80
    return f"{task}\n\n{header}\n{doc_content}\n{footer}"


def run_chat(
    user_proxy: UserProxyAgent,
    manager: GroupChatManager,
    initial_message: str,
) -> None:
    """Trigger the conversation."""
    user_proxy.initiate_chat(
        manager,
        message=initial_message,
    )


def main() -> None:
    args = parse_args()

    if args.env_file.exists():
        load_dotenv(args.env_file)
    # Loading autogen.env if present gives teams a dedicated secret store.
    alt_env = Path("autogen.env")
    if alt_env.exists():
        load_dotenv(alt_env)

    try:
        doc_content = read_document(args.doc)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    try:
        user_proxy, _agents, manager = build_agents(args)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    initial_message = build_initial_message(args.task, doc_content)
    run_chat(user_proxy, manager, initial_message)


if __name__ == "__main__":
    main()
