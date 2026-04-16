import json
import logging
from typing import Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from ..models import ReasoningStep, SourceItem, ToolArtefact

logger = logging.getLogger(__name__)


def extract_sources_from_tool_messages(messages: List) -> List[SourceItem]:
    """Extract sources from tool messages in conversation.

    Parses the JSON output of the retrieve_papers tool to build SourceItem objects.

    :param messages: List of messages from graph state
    :returns: List of SourceItem objects
    """
    sources = []
    seen: set = set()

    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        if getattr(msg, "name", "") != "retrieve_papers":
            continue
        content = msg.content if hasattr(msg, "content") else ""
        if not isinstance(content, str):
            continue
        try:
            data = json.loads(content)
            for doc in data.get("documents", []):
                meta = doc.get("metadata", {})
                arxiv_id = meta.get("arxiv_id", "")
                if not arxiv_id or arxiv_id in seen:
                    continue
                seen.add(arxiv_id)
                authors_raw = meta.get("authors", "")
                if isinstance(authors_raw, list):
                    authors = authors_raw
                else:
                    authors = [a.strip() for a in authors_raw.split(",") if a.strip()]
                sources.append(SourceItem(
                    arxiv_id=arxiv_id,
                    title=meta.get("title", ""),
                    authors=authors,
                    url=meta.get("source", f"https://arxiv.org/abs/{arxiv_id}"),
                    relevance_score=float(meta.get("score", 0.0)),
                ))
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse tool message for sources: {e}")

    return sources


def extract_tool_artefacts(messages: List) -> List[ToolArtefact]:
    """Extract tool artifacts from messages.

    :param messages: List of messages from graph state
    :returns: List of ToolArtefact objects
    """
    artefacts = []

    for msg in messages:
        if isinstance(msg, ToolMessage):
            artefact = ToolArtefact(
                tool_name=getattr(msg, "name", "unknown"),
                tool_call_id=getattr(msg, "tool_call_id", ""),
                content=msg.content,
                metadata={},
            )
            artefacts.append(artefact)

    return artefacts


def create_reasoning_step(
    step_name: str,
    description: str,
    metadata: Optional[Dict] = None,
) -> ReasoningStep:
    """Create a reasoning step record.

    :param step_name: Name of the step/node
    :param description: Human-readable description
    :param metadata: Additional metadata
    :returns: ReasoningStep object
    """
    return ReasoningStep(
        step_name=step_name,
        description=description,
        metadata=metadata or {},
    )


def filter_messages(messages: List) -> List[AIMessage | HumanMessage]:
    """Filter messages to include only HumanMessage and AIMessage types.

    Excludes tool messages and other internal message types.

    :param messages: List of messages to filter
    :returns: Filtered list of messages
    """
    return [msg for msg in messages if isinstance(msg, (HumanMessage, AIMessage))]


def get_latest_query(messages: List) -> str:
    """Get the latest user query from messages.

    :param messages: List of messages
    :returns: Latest query text
    :raises ValueError: If no user query found
    """
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content

    raise ValueError("No user query found in messages")


def get_latest_context(messages: List) -> str:
    """Get the latest context from tool messages.

    Parses the JSON output of retrieve_papers tool into a readable text block.

    :param messages: List of messages
    :returns: Latest context text or empty string
    """
    for msg in reversed(messages):
        if not isinstance(msg, ToolMessage):
            continue
        content = msg.content if hasattr(msg, "content") else ""
        if not isinstance(content, str):
            return ""
        try:
            data = json.loads(content)
            if isinstance(data, dict) and "documents" in data:
                parts = []
                for i, doc in enumerate(data["documents"], 1):
                    text = doc.get("content", "").strip()
                    if not text:
                        continue
                    meta = doc.get("metadata", {})
                    title = meta.get("title", "Unknown")
                    parts.append(f"[{i}] {title}\n{text}")
                return "\n\n".join(parts)
        except (json.JSONDecodeError, KeyError):
            pass
        return content

    return ""
