from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class EvaluationCase:
    item_summary: str
    source_line: int | None = None


@dataclass(slots=True)
class ClarificationTurn:
    question_id: str
    prompt: str
    reason: str
    options: list[dict[str, str]]
    answer: str
    selected_choice_id: str | None
    selected_choice_label: str | None
    source: str


@dataclass(slots=True)
class ShoppingAgentRunRecord:
    design: str
    status_message: str
    profile: dict[str, Any]
    retrieval_batches: list[dict[str, Any]]
    ranked_products: list[dict[str, Any]]
    debug_notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class FinalRecommendationRecord:
    rank: int
    agent_score: int
    agent_rationale: str
    product: dict[str, Any]
    eval_score: int
    eval_rationale: str


@dataclass(slots=True)
class InteractionLogs:
    progress_logs: list[str] = field(default_factory=list)
    clarification_turns: list[ClarificationTurn] = field(default_factory=list)


@dataclass(slots=True)
class EvaluationArtifact:
    item_summary: str
    detailed_item_draft: str
    compressed_prompt: str
    shopping_agent_run: ShoppingAgentRunRecord | None
    interaction_logs: InteractionLogs
    final_recommendations: list[FinalRecommendationRecord]
    timestamps: dict[str, str]
    meta: dict[str, Any]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
