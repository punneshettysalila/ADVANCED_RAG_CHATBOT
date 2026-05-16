"""
Persistent learning loop and knowledge store for the A2A system.

Stores:
- user queries
- agents used
- responses
- verification metadata
- uploaded knowledge chunks

The implementation keeps dependencies light and uses SQLite plus a
portable Python embedding fallback, with optional Hugging Face embeddings
when available.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import re
import sqlite3
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

try:
    from pypdf import PdfReader  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    PdfReader = None

try:
    from huggingface_hub import InferenceClient  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    InferenceClient = None


DEFAULT_DB_FILENAME = "experience_store.sqlite3"
DEFAULT_EMBED_DIM = 256
DEFAULT_EMBED_MODEL = os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


@dataclass
class SearchHit:
    kind: str
    score: float
    title: str
    content: str
    source: str = ""
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "score": round(self.score, 4),
            "title": self.title,
            "content": self.content,
            "source": self.source,
            "metadata": self.metadata or {},
        }


class ExperienceStore:
    """SQLite-backed memory and grounding store."""

    def __init__(
        self,
        base_dir: Optional[str] = None,
        hf_token: Optional[str] = None,
        embed_model: Optional[str] = None,
    ):
        self.base_dir = Path(base_dir or Path(__file__).resolve().parent / ".a2a_memory")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir = self.base_dir / "uploads"
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.base_dir / DEFAULT_DB_FILENAME
        self.hf_token = hf_token or os.getenv("HF_TOKEN", "")
        self.embed_model = embed_model or DEFAULT_EMBED_MODEL
        self.embed_dim = DEFAULT_EMBED_DIM
        self._hf_client = None
        self._init_db()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS experiences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT,
                    session_id TEXT,
                    query TEXT NOT NULL,
                    agents_used TEXT NOT NULL,
                    response TEXT NOT NULL,
                    plan TEXT,
                    verification TEXT,
                    metadata TEXT,
                    embedding TEXT,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_name TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    mime_type TEXT,
                    content TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    metadata TEXT,
                    embedding TEXT,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_experiences_created_at ON experiences(created_at DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_documents_source_name ON documents(source_name)"
            )

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def _get_hf_client(self):
        if InferenceClient is None:
            return None
        if self._hf_client is None:
            try:
                self._hf_client = InferenceClient(api_key=self.hf_token or None, timeout=120)
            except Exception:
                self._hf_client = None
        return self._hf_client

    def _local_embedding(self, text: str) -> List[float]:
        vector = [0.0] * self.embed_dim
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        if not tokens:
            return vector
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            for i in range(0, 16, 2):
                slot = digest[i] % self.embed_dim
                value = (digest[i + 1] / 255.0) + 0.25
                vector[slot] += value
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def embed_text(self, text: str) -> List[float]:
        text = text.strip()
        if not text:
            return [0.0] * self.embed_dim

        client = self._get_hf_client()
        if client is not None:
            try:
                embed = client.feature_extraction(text, model=self.embed_model)
                if isinstance(embed, list) and embed:
                    if isinstance(embed[0], list):
                        embed = embed[0]
                    vector = [float(x) for x in embed]
                    if vector:
                        return vector
            except Exception:
                pass
        return self._local_embedding(text)

    @staticmethod
    def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
        if not left or not right:
            return 0.0
        length = min(len(left), len(right))
        left = left[:length]
        right = right[:length]
        numerator = sum(l * r for l, r in zip(left, right))
        left_norm = math.sqrt(sum(l * l for l in left))
        right_norm = math.sqrt(sum(r * r for r in right))
        if not left_norm or not right_norm:
            return 0.0
        return numerator / (left_norm * right_norm)

    @staticmethod
    def _json_dump(value: Any) -> str:
        return json.dumps(value, ensure_ascii=True)

    @staticmethod
    def _json_load(value: Optional[str], default: Any) -> Any:
        if not value:
            return default
        try:
            return json.loads(value)
        except Exception:
            return default

    # ------------------------------------------------------------------
    # Text chunking / ingest
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"\s+", " ", text or "").strip()

    def chunk_text(self, text: str, chunk_size: int = 1200, overlap: int = 160) -> List[str]:
        text = self._normalize_text(text)
        if not text:
            return []
        if len(text) <= chunk_size:
            return [text]
        chunks: List[str] = []
        start = 0
        while start < len(text):
            end = min(len(text), start + chunk_size)
            chunks.append(text[start:end].strip())
            if end == len(text):
                break
            start = max(end - overlap, start + 1)
        return [chunk for chunk in chunks if chunk]

    def extract_text_from_upload(self, file_name: str, content: bytes, mime_type: str = "") -> str:
        lower = file_name.lower()
        mime_type = (mime_type or "").lower()

        if lower.endswith(".pdf") or "pdf" in mime_type:
            if PdfReader is None:
                return content.decode("utf-8", errors="ignore")
            try:
                reader = PdfReader(io.BytesIO(content))
                pages = []
                for page in reader.pages:
                    extracted = page.extract_text() or ""
                    if extracted.strip():
                        pages.append(extracted)
                return "\n".join(pages)
            except Exception:
                return content.decode("utf-8", errors="ignore")

        if lower.endswith(".csv"):
            try:
                decoded = content.decode("utf-8", errors="ignore")
                reader = csv.reader(io.StringIO(decoded))
                rows = []
                for idx, row in enumerate(reader):
                    rows.append(" | ".join(row))
                    if idx >= 200:
                        break
                return "\n".join(rows)
            except Exception:
                return content.decode("utf-8", errors="ignore")

        if lower.endswith(".json") or lower.endswith(".jsonl"):
            try:
                decoded = content.decode("utf-8", errors="ignore")
                if lower.endswith(".jsonl"):
                    lines = []
                    for row in decoded.splitlines():
                        if row.strip():
                            lines.append(row.strip())
                    return "\n".join(lines)
                return json.dumps(json.loads(decoded), ensure_ascii=True, indent=2)
            except Exception:
                return content.decode("utf-8", errors="ignore")

        return content.decode("utf-8", errors="ignore")

    def ingest_file(self, file_name: str, content: bytes, mime_type: str = "", metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        metadata = metadata or {}
        source_name = metadata.get("source_name") or file_name
        stored_path = self.upload_dir / file_name
        stored_path.write_bytes(content)

        text = self.extract_text_from_upload(file_name=file_name, content=content, mime_type=mime_type)
        chunks = self.chunk_text(text)
        inserted = self.add_document_chunks(
            source_name=source_name,
            file_name=file_name,
            mime_type=mime_type,
            chunks=chunks,
            metadata={**metadata, "stored_path": str(stored_path)},
        )
        return {
            "file_name": file_name,
            "source_name": source_name,
            "chunks_indexed": inserted,
            "stored_path": str(stored_path),
        }

    def add_document_chunks(
        self,
        source_name: str,
        file_name: str,
        mime_type: str,
        chunks: Sequence[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        metadata = metadata or {}
        now = time.time()
        inserted = 0
        with self._connect() as conn:
            for idx, chunk in enumerate(chunks):
                chunk = self._normalize_text(chunk)
                if not chunk:
                    continue
                embedding = self.embed_text(chunk)
                conn.execute(
                    """
                    INSERT INTO documents (
                        source_name, file_name, mime_type, content, chunk_index,
                        metadata, embedding, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        source_name,
                        file_name,
                        mime_type,
                        chunk,
                        idx,
                        self._json_dump(metadata),
                        self._json_dump(embedding),
                        now,
                    ),
                )
                inserted += 1
        return inserted

    # ------------------------------------------------------------------
    # Experience loop
    # ------------------------------------------------------------------

    def record_experience(
        self,
        query: str,
        agents_used: Sequence[str],
        response: str,
        task_id: Optional[str] = None,
        session_id: Optional[str] = None,
        plan: Optional[Dict[str, Any]] = None,
        verification: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        metadata = metadata or {}
        created_at = time.time()
        payload = {
            "task_id": task_id,
            "session_id": session_id,
            "query": query,
            "agents_used": list(agents_used),
            "response": response,
            "plan": plan or {},
            "verification": verification or {},
            "metadata": metadata,
            "created_at": created_at,
        }
        embedding = self.embed_text(f"Q: {query}\nA: {response}")
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO experiences (
                    task_id, session_id, query, agents_used, response,
                    plan, verification, metadata, embedding, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    session_id,
                    query,
                    self._json_dump(list(agents_used)),
                    response,
                    self._json_dump(plan or {}),
                    self._json_dump(verification or {}),
                    self._json_dump(metadata),
                    self._json_dump(embedding),
                    created_at,
                ),
            )
            return int(cursor.lastrowid)

    def _fetch_all_experiences(self, limit: int = 1000) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM experiences ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def search_experiences(self, query: str, limit: int = 3) -> List[SearchHit]:
        needle = self.embed_text(query)
        hits: List[SearchHit] = []
        for row in self._fetch_all_experiences(limit=250):
            score = self._cosine_similarity(needle, self._json_load(row.get("embedding"), []))
            if score <= 0:
                continue
            hits.append(
                SearchHit(
                    kind="experience",
                    score=score,
                    title=row.get("query", "Past task"),
                    content=row.get("response", ""),
                    source=row.get("task_id", ""),
                    metadata={
                        "agents_used": self._json_load(row.get("agents_used"), []),
                        "verification": self._json_load(row.get("verification"), {}),
                        "plan": self._json_load(row.get("plan"), {}),
                        "session_id": row.get("session_id", ""),
                        "created_at": row.get("created_at", 0.0),
                    },
                )
            )
        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[:limit]

    def search_documents(self, query: str, limit: int = 5) -> List[SearchHit]:
        needle = self.embed_text(query)
        hits: List[SearchHit] = []
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM documents ORDER BY created_at DESC LIMIT 1500"
            ).fetchall()
        for row in rows:
            score = self._cosine_similarity(needle, self._json_load(row["embedding"], []))
            if score <= 0:
                continue
            hits.append(
                SearchHit(
                    kind="document",
                    score=score,
                    title=row["file_name"],
                    content=row["content"],
                    source=row["source_name"],
                    metadata={
                        "mime_type": row.get("mime_type", ""),
                        "chunk_index": row.get("chunk_index", 0),
                        "created_at": row.get("created_at", 0.0),
                    },
                )
            )
        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[:limit]

    def search_knowledge(self, query: str, limit: int = 5) -> List[SearchHit]:
        document_hits = self.search_documents(query, limit=limit)
        experience_hits = self.search_experiences(query, limit=max(1, limit // 2))
        combined = document_hits + experience_hits
        combined.sort(key=lambda item: item.score, reverse=True)
        return combined[:limit]

    def summarize_memory_context(self, query: str, limit: int = 3) -> str:
        hits = self.search_knowledge(query, limit=limit)
        if not hits:
            return ""
        lines = ["Relevant memory/context:"]
        for idx, hit in enumerate(hits, start=1):
            preview = self._normalize_text(hit.content)[:260]
            lines.append(f"{idx}. [{hit.kind}] {hit.title}: {preview}")
        return "\n".join(lines)

    def stats(self) -> Dict[str, Any]:
        with self._connect() as conn:
            exp_count = conn.execute("SELECT COUNT(*) AS count FROM experiences").fetchone()["count"]
            doc_count = conn.execute("SELECT COUNT(*) AS count FROM documents").fetchone()["count"]
        return {
            "experiences": int(exp_count),
            "document_chunks": int(doc_count),
            "db_path": str(self.db_path),
        }
