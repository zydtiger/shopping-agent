from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class RankingDesign(StrEnum):
    DIRECT_JSON = "direct_json"
    RAG = "rag"

    @property
    def label(self) -> str:
        return {
            RankingDesign.DIRECT_JSON: "Direct JSON Ranking",
            RankingDesign.RAG: "RAG Candidate Retrieval",
        }[self]


@dataclass(slots=True)
class Product:
    title: str
    price: float
    currency: str
    source_site: str
    product_url: str
    short_description: str
    category: str
    brand: str | None = None
    material: str | None = None
    rating: float | None = None
    review_count: int | None = None
    raw_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Product:
        return cls(
            title=str(payload.get("title", "")),
            price=float(payload.get("price", 0.0)),
            currency=str(payload.get("currency", "USD")),
            source_site=str(payload.get("source_site", "")),
            product_url=str(payload.get("product_url", "")),
            short_description=str(payload.get("short_description", "")),
            category=str(payload.get("category", "")),
            brand=_optional_string(payload.get("brand")),
            material=_optional_string(payload.get("material")),
            rating=_optional_float(payload.get("rating")),
            review_count=_optional_int(payload.get("review_count")),
            raw_metadata=_optional_dict(payload.get("raw_metadata")),
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "latency_ms": self.latency_ms,
            "products": [product.to_dict() for product in self.products],
        }


@dataclass(slots=True)
class RankedProduct:
    rank: int
    score: float
    rationale: str
    product: Product

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "score": round(self.score, 3),
            "rationale": self.rationale,
            "product": self.product.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RankedProduct:
        product_payload = payload.get("product")
        if not isinstance(product_payload, dict):
            raise ValueError(
                "Ranked product payload is missing a valid product object."
            )
        return cls(
            rank=int(payload.get("rank", 0)),
            score=float(payload.get("score", 0.0)),
            rationale=str(payload.get("rationale", "")),
            product=Product.from_dict(product_payload),
        )


@dataclass(slots=True)
class SearchResponse:
    query: str
    design: RankingDesign
    profile: UserPreferenceProfile
    stage: str
    status_message: str
    questions: list[ClarificationQuestion] = field(default_factory=list)
    retrieval_batches: list[RetrievalBatch] = field(default_factory=list)
    ranked_products: list[RankedProduct] = field(default_factory=list)
    debug_notes: list[str] = field(default_factory=list)

    @property
    def requires_clarification(self) -> bool:
        return self.stage == "clarification"


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    return int(value)


def _optional_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}
