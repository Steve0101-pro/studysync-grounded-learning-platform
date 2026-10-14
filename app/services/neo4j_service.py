"""Optional Neo4j knowledge-graph integration for course concepts."""

from __future__ import annotations

import logging
import ssl
from threading import RLock

import certifi

from app.config import settings
from app.services.study_materials import tokenize

logger = logging.getLogger(__name__)
_driver = None
_lock = RLock()


def _get_driver():
    global _driver
    uri = settings.NEO4J_URI.strip()
    if not uri:
        return None
    with _lock:
        if _driver is None:
            from neo4j import GraphDatabase

            username = settings.NEO4J_USERNAME.strip()
            password = settings.NEO4J_PASSWORD.get_secret_value()
            if not username or not password:
                raise RuntimeError(
                    "NEO4J_USERNAME and NEO4J_PASSWORD are required when NEO4J_URI is set"
                )

            driver_options = {}
            if uri.startswith(("neo4j+s://", "bolt+s://")):
                ssl_context = ssl.create_default_context()
                ssl_context.load_verify_locations(cafile=certifi.where())
                scheme, address = uri.split("://", 1)
                uri = f"{scheme[:-2]}://{address}"
                driver_options = {
                    "encrypted": True,
                    "ssl_context": ssl_context,
                }

            _driver = GraphDatabase.driver(
                uri,
                auth=(username, password),
                **driver_options,
            )
        return _driver


_STOP_WORDS = {
    "about", "after", "again", "also", "among", "because", "before", "being",
    "between", "could", "does", "during", "each", "from", "have", "into",
    "more", "most", "other", "over", "same", "some", "such", "than", "that",
    "their", "there", "these", "they", "this", "those", "through", "under",
    "very", "were", "when", "where", "which", "while", "with", "would",
}


def _concepts(content: str, limit: int = 10) -> list[str]:
    counts: dict[str, int] = {}
    for term in tokenize(content):
        if term not in _STOP_WORDS:
            counts[term] = counts.get(term, 0) + 1
    return sorted(counts, key=lambda term: (-counts[term], term))[:limit]


def index_source_passages(
    *,
    user_id: str,
    source_id: str,
    filename: str,
    passages: list[dict],
) -> bool:
    """Persist source, passage locations, and extracted topic terms when configured."""
    driver = _get_driver()
    if driver is None:
        return False
    records = [
        {
            "location_type": passage["location_type"],
            "location_number": passage["location_number"],
            "chunk_index": passage["chunk_index"],
            "concepts": _concepts(passage["content"]),
        }
        for passage in passages
    ]
    with driver.session() as session:
        session.run(
            """
            MERGE (s:StudySource {id: $source_id, user_id: $user_id})
            SET s.filename = $filename
            WITH s
            UNWIND $passages AS passage
            MERGE (p:Passage {
                source_id: $source_id,
                location_type: passage.location_type,
                location_number: passage.location_number,
                chunk_index: passage.chunk_index
            })
            MERGE (s)-[:HAS_PASSAGE]->(p)
            WITH p, passage
            UNWIND passage.concepts AS concept_name
            MERGE (c:Concept {name: concept_name, user_id: $user_id})
            MERGE (p)-[:MENTIONS]->(c)
            """,
            user_id=user_id,
            source_id=source_id,
            filename=filename,
            passages=records,
        )
    return True


def close_neo4j() -> None:
    global _driver
    with _lock:
        if _driver is not None:
            _driver.close()
            _driver = None


def neo4j_health() -> dict[str, str | bool]:
    try:
        driver = _get_driver()
    except Exception as exc:
        logger.warning("Neo4j driver configuration failed: %s", type(exc).__name__)
        return {"configured": True, "status": "unhealthy"}
    if driver is None:
        return {"configured": False, "status": "disabled"}
    try:
        driver.verify_connectivity()
        return {"configured": True, "status": "healthy"}
    except Exception as exc:
        logger.warning("Neo4j health check failed: %s", type(exc).__name__)
        return {"configured": True, "status": "unhealthy"}
