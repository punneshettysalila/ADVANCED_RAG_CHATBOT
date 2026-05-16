"""
Small compatibility layer for internal knowledge retrieval.

This wraps ExperienceStore so upload/search endpoints and orchestrators
have a clearer semantic entry point for grounding.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from experience_store import ExperienceStore, SearchHit


class KnowledgeBase:
    def __init__(self, store: Optional[ExperienceStore] = None):
        self.store = store or ExperienceStore()

    def ingest_file(self, file_name: str, content: bytes, mime_type: str = "", metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.store.ingest_file(file_name=file_name, content=content, mime_type=mime_type, metadata=metadata)

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        hits = self.store.search_knowledge(query, limit=limit)
        return [hit.to_dict() for hit in hits]

    def context_block(self, query: str, limit: int = 3) -> str:
        return self.store.summarize_memory_context(query, limit=limit)
