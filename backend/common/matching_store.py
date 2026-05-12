import logging
from typing import List, Dict, Any, Optional
from backend.config import settings, EMBEDDING_CONFIGS
from backend.standard.session_store_v2 import hydrate_session, destroy_session, query_session, create_session

logger = logging.getLogger(__name__)


def create_matching_triple_store(
    session_id: str,
    records: Optional[List[Dict[str, Any]]],
    records2: Optional[List[Dict[str, Any]]],
    matchrecords: Optional[List[Dict[str, Any]]],
    predictions: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, int]:
    """
    Creates three separate in-memory vector stores for a matching session.
    Each sub-store gets its own .pkl file (session_id_r1.pkl, _r2.pkl, _mr.pkl).
    Embeds on-the-fly from FptEnglish.db.
    """
    from backend.common.utils import get_embedding
    from langchain_core.documents import Document
    from backend.common.parser import parse_compressed_records

    emb_model = settings.EMBEDDING_MODEL
    emb_cfg = EMBEDDING_CONFIGS.get(emb_model, {"dim": settings.EMBEDDING_DIM})
    dim = emb_cfg["dim"]
    embedding = get_embedding(emb_model)

    results = {"records": 0, "records2": 0, "matchrecords": 0}

    datasets = [
        ("r1", records),
        ("r2", records2),
        ("mr", matchrecords)
    ]

    for suffix, data in datasets:
        sub_id = f"{session_id}_{suffix}"
        destroy_session(sub_id)

        if not data:
            continue

        dataset_refs = parse_compressed_records(data)

        loaded = hydrate_session(
            session_id=sub_id,
            dataset_refs=dataset_refs,
            source_db_path=settings.SOURCE_DB_PATH,
            embedding=embedding,
            dim=dim
        )

        mapping = {"r1": "records", "r2": "records2", "mr": "matchrecords"}
        results[mapping[suffix]] = loaded

    if predictions:
        docs_to_index = []
        for item in predictions:
            tbl = item.get("table", "Unknown")
            keys = ", ".join(item.get("keys", []))
            for text in item.get("predictions", []):
                if not text.strip(): continue
                content = f"Table: {tbl} | Keywords: {keys} | Analysis: {text}"
                docs_to_index.append(Document(
                    page_content=content,
                    metadata={"table": tbl, "source": "prediction", "is_prediction": True}
                ))

        if docs_to_index:
            sub_id = f"{session_id}_mr"
            total_mr = create_session(
                session_id=sub_id,
                documents=docs_to_index,
                embedding=embedding,
                dim=dim,
                append=True
            )
            results["matchrecords"] = total_mr

    logger.info(f"Matching Triple Store Created [Session: {session_id}]: {results}")
    return results


def query_matching_triple(
    session_id: str,
    query: str,
    k_per_store: int = 5,
    filter_stores: List[str] = None
) -> Dict[str, List[Any]]:
    from backend.common.utils import get_embedding
    emb_model = settings.EMBEDDING_MODEL
    emb_fn = get_embedding(emb_model)
    results = {"records": [], "records2": [], "matchrecords": []}
    store_map = {"records": "r1", "records2": "r2", "matchrecords": "mr"}
    to_query = filter_stores if filter_stores else list(store_map.keys())

    for key in to_query:
        suffix = store_map.get(key)
        if not suffix: continue
        sub_id = f"{session_id}_{suffix}"
        try:
            docs = query_session(sub_id, query, emb_fn, k=k_per_store)
            results[key] = [
                {
                    "content": d.page_content,
                    "table": d.metadata.get("table"),
                    "score": d.metadata.get("score")
                }
                for d in docs
            ]
        except Exception as e:
            logger.warning(f"Failed to query matching sub-store {sub_id}: {e}")
    return results
