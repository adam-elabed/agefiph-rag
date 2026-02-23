import base64, re
from azure.search.documents.models import VectorizedQuery

def decode_parent_id(parent_id: str) -> str:
    if not parent_id:
        return ""
    try:
        s = parent_id.strip()
        s += "=" * (-len(s) % 4)
        url = base64.urlsafe_b64decode(s.encode("utf-8")).decode("utf-8")
        # sanitize
        url = re.sub(r"\.pdf\d+\b", ".pdf", url)
        url = re.sub(r"\.pdf\d+_", ".pdf_", url)
        return url
    except Exception:
        return parent_id

async def retrieve(search_client, embedder, query: str, k: int = 8):
    qvec = (await embedder.generate_embeddings([query]))[0]
    vq = VectorizedQuery(vector=qvec, k=k, fields="text_vector")

    results = await search_client.search(
        search_text=query,              # hybrid
        vector_queries=[vq],
        select=["chunk", "title", "parent_id", "chunk_id"],
        top=k,
    )

    rows = []
    async for r in results:
        chunk = (r.get("chunk") or "").strip()
        if not chunk or "SOMMAIRE" in chunk:
            continue

        rows.append({
            "text": chunk,
            "title": r.get("title") or "",
            "chunk_id": r.get("chunk_id") or "",
            "source_url": decode_parent_id(r.get("parent_id") or ""),
            "score": float(r.get("@search.score", 0.0)),
        })

    rows.sort(key=lambda x: x["score"], reverse=True)

    # Gate: si hors-sujet (ex: "hello"), renvoie rien
    if not rows or rows[0]["score"] < 0.30:
        return []

    # Dedup
    seen = set()
    out = []
    for row in rows:
        if row["text"] in seen:
            continue
        seen.add(row["text"])
        out.append(row)
        if len(out) == 4:
            break

    return out
