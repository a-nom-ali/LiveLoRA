"""Load and parse OpenAI ChatGPT export data for use as test conversations.

OpenAI's export (Settings → Data controls → Export data) produces a zip
containing `conversations.json`. The format uses a tree structure (not a
flat list) where each message node has a parent pointer.

This module extracts clean (user, assistant) turn pairs suitable for
testing LiveLoRA's TTT loop.
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Turn:
    """A single message in a conversation."""

    role: str  # "user", "assistant", "system", "tool"
    content: str
    model: str | None = None  # e.g. "gpt-4o", "gpt-4", etc.
    timestamp: float | None = None


@dataclass
class Conversation:
    """A parsed conversation with flat turn list."""

    id: str
    title: str
    turns: list[Turn] = field(default_factory=list)
    create_time: float | None = None
    model: str | None = None  # primary model used

    @property
    def user_turns(self) -> list[Turn]:
        return [t for t in self.turns if t.role == "user"]

    @property
    def assistant_turns(self) -> list[Turn]:
        return [t for t in self.turns if t.role == "assistant"]

    @property
    def turn_pairs(self) -> list[tuple[Turn, Turn]]:
        """Extract (user, assistant) pairs for testing."""
        pairs = []
        for i, turn in enumerate(self.turns):
            if turn.role == "user" and i + 1 < len(self.turns):
                next_turn = self.turns[i + 1]
                if next_turn.role == "assistant" and next_turn.content:
                    pairs.append((turn, next_turn))
        return pairs


def _extract_content(message: dict) -> str:
    """Extract text content from a message node, handling various content types."""
    content = message.get("content", {})
    if not content:
        return ""

    parts = content.get("parts", [])
    if not parts:
        return ""

    text_parts = []
    for part in parts:
        if isinstance(part, str):
            text_parts.append(part)
        elif isinstance(part, dict):
            # Multimodal content (images, code, etc.) — extract text if present
            if "text" in part:
                text_parts.append(part["text"])

    return "\n".join(text_parts).strip()


def _flatten_tree(mapping: dict) -> list[Turn]:
    """Walk the conversation tree and produce a flat chronological turn list.

    The mapping is a dict of node_id -> node, where each node has:
    - "parent": parent node_id (or None for root)
    - "children": list of child node_ids
    - "message": the message object (or None for the root node)

    We walk from root to leaves following the first child at each branch
    (OpenAI's default conversation path).
    """
    if not mapping:
        return []

    # Find root node (parent is None or parent not in mapping)
    root_id = None
    for node_id, node in mapping.items():
        parent = node.get("parent")
        if parent is None or parent not in mapping:
            root_id = node_id
            break

    if root_id is None:
        return []

    # Walk the tree depth-first, following first child at each level
    turns = []
    current_id = root_id
    visited = set()

    while current_id and current_id not in visited:
        visited.add(current_id)
        node = mapping.get(current_id)
        if node is None:
            break

        message = node.get("message")
        if message and message.get("author", {}).get("role") in ("user", "assistant", "system"):
            content = _extract_content(message)
            if content:
                role = message["author"]["role"]
                model = message.get("metadata", {}).get("model_slug")
                timestamp = message.get("create_time")
                turns.append(Turn(role=role, content=content, model=model, timestamp=timestamp))

        # Follow children — take the last child (OpenAI uses last = active branch)
        children = node.get("children", [])
        current_id = children[-1] if children else None

    return turns


def load_conversations_json(path: str | Path) -> list[Conversation]:
    """Load conversations from a conversations.json file.

    Args:
        path: Path to conversations.json (or the zip file from OpenAI export).

    Returns:
        List of parsed Conversation objects, sorted by creation time.
    """
    path = Path(path)

    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            # Find conversations.json in the zip
            json_files = [f for f in zf.namelist() if f.endswith("conversations.json")]
            if not json_files:
                raise FileNotFoundError("No conversations.json found in zip")
            with zf.open(json_files[0]) as f:
                raw = json.load(f)
    else:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)

    conversations = []
    for conv_data in raw:
        mapping = conv_data.get("mapping", {})
        turns = _flatten_tree(mapping)

        if not turns:
            continue

        # Determine primary model from assistant turns
        assistant_models = [t.model for t in turns if t.role == "assistant" and t.model]
        primary_model = assistant_models[-1] if assistant_models else None

        conv = Conversation(
            id=conv_data.get("id", ""),
            title=conv_data.get("title", "Untitled"),
            turns=turns,
            create_time=conv_data.get("create_time"),
            model=primary_model,
        )
        conversations.append(conv)

    # Sort by creation time
    conversations.sort(key=lambda c: c.create_time or 0)
    return conversations


def load_turn_pairs(
    path: str | Path,
    min_user_length: int = 10,
    min_assistant_length: int = 20,
    max_pairs: int | None = None,
) -> list[tuple[str, str]]:
    """Convenience: load (user_text, assistant_text) pairs for testing.

    Filters out very short messages and system/tool turns.

    Args:
        path: Path to conversations.json or export zip.
        min_user_length: Minimum character length for user messages.
        min_assistant_length: Minimum character length for assistant responses.
        max_pairs: Maximum number of pairs to return (None = all).

    Returns:
        List of (user_text, assistant_text) string tuples.
    """
    conversations = load_conversations_json(path)
    pairs = []

    for conv in conversations:
        for user_turn, assistant_turn in conv.turn_pairs:
            if len(user_turn.content) >= min_user_length and len(assistant_turn.content) >= min_assistant_length:
                pairs.append((user_turn.content, assistant_turn.content))

    if max_pairs is not None:
        pairs = pairs[:max_pairs]

    return pairs


def print_stats(path: str | Path):
    """Print summary statistics about a ChatGPT export."""
    conversations = load_conversations_json(path)

    total_turns = sum(len(c.turns) for c in conversations)
    total_pairs = sum(len(c.turn_pairs) for c in conversations)
    models = {}
    for c in conversations:
        if c.model:
            models[c.model] = models.get(c.model, 0) + 1

    print(f"Conversations: {len(conversations)}")
    print(f"Total turns:   {total_turns}")
    print(f"Turn pairs:    {total_pairs} (user→assistant)")
    print(f"Models used:   {models}")

    # Length stats
    user_lengths = []
    assistant_lengths = []
    for c in conversations:
        for u, a in c.turn_pairs:
            user_lengths.append(len(u.content))
            assistant_lengths.append(len(a.content))

    if user_lengths:
        print(f"\nUser message length:      median={sorted(user_lengths)[len(user_lengths)//2]}, "
              f"mean={sum(user_lengths)//len(user_lengths)}, max={max(user_lengths)}")
        print(f"Assistant message length: median={sorted(assistant_lengths)[len(assistant_lengths)//2]}, "
              f"mean={sum(assistant_lengths)//len(assistant_lengths)}, max={max(assistant_lengths)}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m livelora.data.chatgpt_loader <path-to-conversations.json-or-zip>")
        sys.exit(1)

    print_stats(sys.argv[1])
