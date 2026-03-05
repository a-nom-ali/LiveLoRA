"""Tests for ChatGPT export data loader."""

import json
import tempfile
from pathlib import Path

from livelora.data.chatgpt_loader import (
    Conversation,
    Turn,
    _extract_content,
    _flatten_tree,
    load_conversations_json,
    load_turn_pairs,
)

# Minimal conversations.json structure matching OpenAI's export format
SAMPLE_EXPORT = [
    {
        "id": "conv-001",
        "title": "Test Conversation",
        "create_time": 1700000000.0,
        "mapping": {
            "root": {
                "id": "root",
                "parent": None,
                "children": ["msg-1"],
                "message": None,
            },
            "msg-1": {
                "id": "msg-1",
                "parent": "root",
                "children": ["msg-2"],
                "message": {
                    "author": {"role": "user"},
                    "content": {"parts": ["What is persistent homology?"]},
                    "create_time": 1700000001.0,
                    "metadata": {},
                },
            },
            "msg-2": {
                "id": "msg-2",
                "parent": "msg-1",
                "children": ["msg-3"],
                "message": {
                    "author": {"role": "assistant"},
                    "content": {
                        "parts": [
                            "Persistent homology is a method in topological data analysis "
                            "that studies the shape of data across multiple scales."
                        ]
                    },
                    "create_time": 1700000002.0,
                    "metadata": {"model_slug": "gpt-4"},
                },
            },
            "msg-3": {
                "id": "msg-3",
                "parent": "msg-2",
                "children": [],
                "message": {
                    "author": {"role": "user"},
                    "content": {"parts": ["Can you give an example?"]},
                    "create_time": 1700000003.0,
                    "metadata": {},
                },
            },
        },
    },
    {
        "id": "conv-002",
        "title": "Short Chat",
        "create_time": 1700001000.0,
        "mapping": {
            "root": {
                "id": "root",
                "parent": None,
                "children": ["m1"],
                "message": None,
            },
            "m1": {
                "id": "m1",
                "parent": "root",
                "children": ["m2"],
                "message": {
                    "author": {"role": "user"},
                    "content": {"parts": ["Hi"]},
                    "create_time": 1700001001.0,
                    "metadata": {},
                },
            },
            "m2": {
                "id": "m2",
                "parent": "m1",
                "children": [],
                "message": {
                    "author": {"role": "assistant"},
                    "content": {"parts": ["Hello! How can I help you today?"]},
                    "create_time": 1700001002.0,
                    "metadata": {"model_slug": "gpt-3.5-turbo"},
                },
            },
        },
    },
]


def _write_sample(tmp_dir: Path) -> Path:
    path = tmp_dir / "conversations.json"
    with open(path, "w") as f:
        json.dump(SAMPLE_EXPORT, f)
    return path


class TestExtractContent:
    def test_string_parts(self):
        msg = {"content": {"parts": ["hello", " world"]}}
        assert _extract_content(msg) == "hello\n world"

    def test_empty_content(self):
        assert _extract_content({}) == ""
        assert _extract_content({"content": {}}) == ""
        assert _extract_content({"content": {"parts": []}}) == ""

    def test_dict_parts_with_text(self):
        msg = {"content": {"parts": [{"text": "code here"}]}}
        assert _extract_content(msg) == "code here"


class TestFlattenTree:
    def test_basic_tree(self):
        mapping = SAMPLE_EXPORT[0]["mapping"]
        turns = _flatten_tree(mapping)
        assert len(turns) == 3
        assert turns[0].role == "user"
        assert turns[1].role == "assistant"
        assert turns[2].role == "user"
        assert "persistent homology" in turns[0].content

    def test_empty_mapping(self):
        assert _flatten_tree({}) == []

    def test_model_extraction(self):
        mapping = SAMPLE_EXPORT[0]["mapping"]
        turns = _flatten_tree(mapping)
        assert turns[1].model == "gpt-4"


class TestLoadConversations:
    def test_load_json(self, tmp_path):
        path = _write_sample(tmp_path)
        convs = load_conversations_json(path)
        assert len(convs) == 2
        assert convs[0].title == "Test Conversation"
        assert convs[1].title == "Short Chat"

    def test_sorted_by_time(self, tmp_path):
        path = _write_sample(tmp_path)
        convs = load_conversations_json(path)
        assert convs[0].create_time < convs[1].create_time

    def test_turn_pairs(self, tmp_path):
        path = _write_sample(tmp_path)
        convs = load_conversations_json(path)
        # First conv has 2 user messages but only 1 user→assistant pair
        # (last user message has no assistant response)
        pairs = convs[0].turn_pairs
        assert len(pairs) == 1
        assert "persistent homology" in pairs[0][0].content

    def test_model_detection(self, tmp_path):
        path = _write_sample(tmp_path)
        convs = load_conversations_json(path)
        assert convs[0].model == "gpt-4"
        assert convs[1].model == "gpt-3.5-turbo"


class TestLoadTurnPairs:
    def test_filters_short_messages(self, tmp_path):
        path = _write_sample(tmp_path)
        # "Hi" is only 2 chars, should be filtered with min_user_length=10
        pairs = load_turn_pairs(path, min_user_length=10, min_assistant_length=20)
        assert len(pairs) == 1
        assert "persistent homology" in pairs[0][0]

    def test_max_pairs(self, tmp_path):
        path = _write_sample(tmp_path)
        pairs = load_turn_pairs(path, min_user_length=1, min_assistant_length=1, max_pairs=1)
        assert len(pairs) == 1
