from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class RankingDesign(StrEnum):
    DIRECT_JSON = "direct_json"
    SQL = "sql"

    @property
    def label(self) -> str:
        return {
            RankingDesign.DIRECT_JSON: "Direct JSON Ranking",
            RankingDesign.SQL: "SQL-backed Ranking",
        }[self]


@dataclass(slots=True)
class Product:
    # NOTE: the following fields are commented out because they are NOT implemented right now,
    # they might be implemented later, but do not touch it now
    title: str
    price: float
    # currency: str
    source_site: str
    product_url: str
    # short_description: str
    # category: str
    # brand: str | None = None
    # material: str | None = None
    rating: float | None = None
    # review_count: int | None = None
    # raw_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Product:
        return cls(
            title=str(payload.get("title", "")),
            price=float(payload.get("price", 0.0)),
            source_site=str(payload.get("source_site", "")),
            product_url=str(payload.get("product_url", "")),
            rating=_optional_float(payload.get("rating")),
        )


@dataclass(slots=True)
class ClarificationOption:
    id: str
    label: str
    description: str


@dataclass(slots=True)
class ClarificationQuestion:
    id: str
    prompt: str
    options: list[ClarificationOption]
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class UserPreferenceProfile:
    raw_query: str
    clarified_answers: dict[str, str] = field(default_factory=dict)
    inferred_requirements: list[str] = field(default_factory=list)
    uncertainty_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RetrievalBatch:
    source: str
    products: list[Product]
    latency_ms: int


@dataclass(slots=True)
class RankedProduct:
    rank: int
    score: int
    rationale: str
    product: Product


@dataclass(slots=True)
class SearchResponse:
    query: str
    design: RankingDesign
    profile: UserPreferenceProfile
    stage: str
    status_message: str
    retrieval_batches: list[RetrievalBatch] = field(default_factory=list)
    ranked_products: list[RankedProduct] = field(default_factory=list)
    debug_notes: list[str] = field(default_factory=list)


def _optional_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    return float(value)
