"""把现有节点的原始结果转换为统一 Evidence。"""

import hashlib
from collections.abc import Mapping
from typing import Any

from processor.query_process.evidence.models import (
    Evidence,
    EvidenceSource,
)
from processor.query_process.tools.contracts import SearchStrategy


def _get_value(obj: Any, key: str, default: Any = None) -> Any:
    """同时兼容普通字典和 pymilvus 实体对象。"""
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _to_float(value: Any) -> float | None:
    """把分数安全转换为浮点数。"""
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _fallback_id(source: str, content: str) -> str:
    """缺少业务 ID 时，使用正文哈希生成稳定标识。"""
    digest = hashlib.sha1(content.encode("utf-8")).hexdigest()[:16]
    return f"{source}:{digest}"


def from_milvus_hit(
    hit: Any,
    strategy: SearchStrategy,
    rank: int,
) -> Evidence:
    """把 Milvus Hit 转换为统一 Evidence。"""

    # chunk_id 用于后续 RRF 去重和结果合并。
    chunk_id = _get_value(hit, "id")

    # Milvus 的业务字段通常放在 entity 中。
    entity = _get_value(hit, "entity", {}) or {}
    content = str(_get_value(entity, "content", "") or "")

    evidence_id = (
        f"local:{chunk_id}"
        if chunk_id is not None
        else _fallback_id("local", content)
    )

    return Evidence(
        evidence_id=evidence_id,
        source=EvidenceSource.LOCAL,
        chunk_id=chunk_id,
        file_title=_get_value(entity, "file_title"),
        title=_get_value(entity, "title"),
        item_name=_get_value(entity, "item_name"),
        content=content,
        score=_to_float(_get_value(hit, "distance")),
        score_type="milvus_distance",
        rank=rank,
        metadata={"strategy": strategy.value},
    )


def from_web_doc(doc: dict[str, Any], rank: int) -> Evidence:
    """把 MCP 网页结果转换为统一 Evidence。"""
    url = str(doc.get("url") or "")
    content = str(doc.get("snippet") or "")

    return Evidence(
        evidence_id=f"web:{url}" if url else _fallback_id("web", content),
        source=EvidenceSource.WEB,
        title=doc.get("title"),
        content=content,
        url=url or None,
        rank=rank,
    )


def from_rrf_chunk(chunk: dict[str, Any], rank: int) -> Evidence:
    """把 RRF 节点输出转换为统一 Evidence。"""
    chunk_id = chunk.get("chunk_id")
    content = str(chunk.get("content") or "")

    return Evidence(
        evidence_id=(
            f"local:{chunk_id}"
            if chunk_id is not None
            else _fallback_id("local", content)
        ),
        source=EvidenceSource.LOCAL,
        chunk_id=chunk_id,
        file_title=chunk.get("file_title"),
        title=chunk.get("title"),
        item_name=chunk.get("item_name"),
        content=content,
        score_type="rrf",
        rank=rank,
    )


def from_rerank_doc(doc: dict[str, Any], rank: int) -> Evidence:
    """把 Reranker 节点输出转换为统一 Evidence。"""
    source = (
        EvidenceSource.WEB
        if doc.get("source") == "web"
        else EvidenceSource.LOCAL
    )
    content = str(doc.get("content") or "")
    chunk_id = doc.get("chunk_id")

    return Evidence(
        evidence_id=(
            f"{source.value}:{chunk_id}"
            if chunk_id is not None
            else _fallback_id(source.value, content)
        ),
        source=source,
        chunk_id=chunk_id,
        title=doc.get("title"),
        content=content,
        url=doc.get("url"),
        score=_to_float(doc.get("score")),
        score_type="rerank",
        rank=rank,
    )
