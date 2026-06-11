"""Helper for working with the persistent Chroma vector store."""
from __future__ import annotations

from pathlib import Path
from threading import RLock
from typing import Any, Dict, Iterable, List, Sequence

try:
    import chromadb
    from chromadb.api.models.Collection import Collection
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "chromadb is required for UdaciScan. Install via pip install chromadb"
    ) from exc

_CLIENT_CACHE: Dict[str, "chromadb.PersistentClient"] = {}
_COL_CACHE: Dict[tuple[str, str], Collection] = {}
_LOCK = RLock()


def _normalize_path(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve())


def _get_client(path: str | Path) -> "chromadb.PersistentClient":
    norm = _normalize_path(path)
    with _LOCK:
        if norm in _CLIENT_CACHE:
            return _CLIENT_CACHE[norm]
        print(f"[UdaciScan] Loading Chroma vector store at {norm}...", flush=True)
        Path(norm).mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=norm)
        print("[UdaciScan] Chroma ready.", flush=True)
        _CLIENT_CACHE[norm] = client
        return client


def get_vs(path: str | Path, collection: str) -> Collection:
    """Return (and cache) a Chroma collection."""
    key = (_normalize_path(path), collection)
    with _LOCK:
        if key in _COL_CACHE:
            return _COL_CACHE[key]
        client = _get_client(path)
        try:
            col = client.get_collection(collection)
        except Exception:
            col = client.create_collection(collection)
        _COL_CACHE[key] = col
        return col


def upsert_documents(
    collection: Collection,
    *,
    ids: Sequence[str],
    documents: Sequence[str],
    metadatas: Sequence[Dict[str, Any]],
    embeddings: Sequence[Sequence[float]] | None = None,
) -> None:
    """Upsert payloads into the provided collection."""
    if len(ids) != len(documents) or len(ids) != len(metadatas):
        raise ValueError("ids, documents, and metadatas must have equal length")
    payload: Dict[str, Any] = {
        "ids": list(ids),
        "documents": list(documents),
        "metadatas": list(metadatas),
    }
    if embeddings is not None:
        payload["embeddings"] = list(embeddings)
    collection.upsert(**payload)


def get_existing_ids(collection: Collection) -> List[str]:
    """Return all IDs currently present in the collection."""
    ids: List[str] = []
    cursor = None
    while True:
        batch = collection.get(ids=None, where=None, include=["metadatas"], limit=1000, offset=cursor or 0)
        fetched = batch.get("ids", [])
        if not fetched:
            break
        ids.extend(fetched)
        cursor = (cursor or 0) + len(fetched)
        if len(fetched) < 1000:
            break
    return ids