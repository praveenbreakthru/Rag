"""
Hybrid retrieval engine combining:
1. pgvector semantic cosine similarity
2. PostgreSQL tsvector full-text search (ts_rank)
3. Reciprocal Rank Fusion (RRF)
4. Financial metadata and period filtering
"""

import logging
from typing import Optional, Any
from ragcli.config import settings
from ragcli.database import get_db_cursor
from ragcli.schemas import DocumentChunk
from ragcli.ollama_client import OllamaClient
from ragcli.text_normalizer import extract_financial_periods

logger = logging.getLogger(__name__)


class BGEReranker:
    """Singleton wrapper for local BAAI/bge-reranker-base model."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._model = None
            cls._instance._tokenizer = None
        return cls._instance

    def _load_model(self):
        if self._model is None:
            try:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                import os

                path = settings.reranker_path
                if not os.path.exists(path):
                    path = settings.reranker_model

                logger.info(f"Loading local reranker model from: {path}")
                self._tokenizer = AutoTokenizer.from_pretrained(path)
                self._model = AutoModelForSequenceClassification.from_pretrained(path)
                self._model.eval()
            except Exception as e:
                logger.warning(f"Could not load BGE reranker: {e}")
                self._model = False

    def rerank(self, query: str, chunks: list[DocumentChunk], top_k: int) -> list[DocumentChunk]:
        """Rerank retrieved candidate chunks using BGE Cross-Encoder."""
        if not chunks:
            return []
        self._load_model()
        if not self._model or not self._tokenizer:
            return chunks[:top_k]

        try:
            import torch

            pairs = [[query, c.source_text] for c in chunks]
            with torch.no_grad():
                inputs = self._tokenizer(pairs, padding=True, truncation=True, return_tensors="pt", max_length=512)
                logits = self._model(**inputs, return_dict=True).logits.view(-1).float()
                scores = logits.tolist()

            # Pair chunks with scores
            chunk_scores = list(zip(chunks, scores))
            # Sort descending by reranker score
            chunk_scores.sort(key=lambda x: x[1], reverse=True)
            for c, sc in chunk_scores:
                c.rank_score = float(sc)
            return [c for c, _ in chunk_scores[:top_k]]
        except Exception as e:
            logger.warning(f"Reranking error, falling back to initial ranks: {e}")
            return chunks[:top_k]


class HybridRetriever:
    def __init__(self):
        self.ollama = OllamaClient()
        self.reranker = BGEReranker() if getattr(settings, "use_reranker", True) else None

    def search_vector(
        self,
        query_embedding: list[float],
        top_k: int = settings.retrieval_top_k,
        financial_period: Optional[str] = None,
    ) -> list[tuple[str, float]]:
        """Vector similarity search using pgvector cosine distance (<=>). Returns (chunk_id, similarity)."""
        sql = """
            SELECT c.id, 1 - (e.embedding <=> %s::vector) AS similarity
            FROM document_chunks c
            JOIN chunk_embeddings e ON c.id = e.chunk_id
        """
        params: list[Any] = [query_embedding]

        if financial_period:
            sql += " WHERE c.financial_period ILIKE %s"
            params.append(f"%{financial_period}%")

        sql += " ORDER BY e.embedding <=> %s::vector ASC LIMIT %s;"
        params.extend([query_embedding, top_k])

        with get_db_cursor() as cur:
            cur.execute(sql, tuple(params))
            return [(str(row[0]), float(row[1])) for row in cur.fetchall()]

    def search_fulltext(
        self,
        query_text: str,
        top_k: int = settings.retrieval_top_k,
        financial_period: Optional[str] = None,
    ) -> list[tuple[str, float]]:
        """Full-text search using PostgreSQL ts_rank with AND phrase prioritization and OR fallback."""
        import re
        STOP_WORDS = {"what", "was", "the", "of", "as", "at", "in", "for", "to", "and", "a", "an", "is", "are", "on", "with", "from", "by", "that", "this", "tell", "me", "show", "who", "give"}
        COMPANY_BOILERPLATE = {"sakthi", "finance", "limited"}
        all_words = [w.lower() for w in re.findall(r"\b[a-zA-Z0-9_-]+\b", query_text)]
        content_words = [w for w in all_words if w not in STOP_WORDS]
        # Informative words excluding boilerplate company tokens to avoid drowning out specific queries
        informative_words = [w for w in content_words if w not in COMPANY_BOILERPLATE]
        if not informative_words:
            informative_words = content_words

        # Prioritize key topic words
        KEY_FIN_TERMS = {
            "tangible", "asset", "assets", "maturity", "carrying", "revenue", "profit",
            "liability", "liabilities", "depreciation", "borrowing", "borrowings",
            "equity", "debt", "deposit", "deposits", "interest", "cost", "costs",
            "obligation", "benefit", "employee", "finance", "expense", "expenses",
            "tax", "dividend", "repair", "repairs", "maintenance", "audit", "fee", "fees",
            "rent", "rates", "legal", "professional", "travelling", "conveyance", "power", "fuel",
            "director", "directors", "board", "phone", "email", "mail", "contact", "address",
            "branch", "branches", "network", "networks", "ebitda"
        }
        topic_words = [w for w in informative_words if w in KEY_FIN_TERMS or any(k in w for k in KEY_FIN_TERMS)]

        results_map: dict[str, float] = {}

        # 1. Natural phrase matching using phraseto_tsquery (handles stop words naturally, e.g. "Repairs and Maintenance")
        GENERIC_STOP = STOP_WORDS | COMPANY_BOILERPLATE | {
            "year", "ended", "march", "31st", "2019", "2020", "2021", "2022", "2023", "2024", "2025", "2026", "period", "financial",
            "as", "at", "total", "end", "to", "of", "in", "and", "or", "the", "for", "from"
        }
        clean_words = [w for w in re.findall(r"\b[a-zA-Z0-9_-]+\b", query_text)]
        phrase_candidates = []

        # Check for explicit Note queries (e.g., 'Note 4', 'Note 17', 'Notes 3A')
        note_match = re.search(r"(?i)\bnotes?\s*(?:no\.?|number)?\s*(\d{1,2}[A-Za-z]?)\b", query_text)
        if note_match:
            n_num = note_match.group(1).upper()
            phrase_candidates.append(f"NOTE {n_num}")
            phrase_candidates.append(f"Note {n_num}")

        # Prioritize 2-word, 3-word, and 4-word core noun phrases
        # Only accept candidate phrases that contain at least one non-generic informative word (e.g. assets, liabilities, revenue)
        for n in (2, 3, 4):
            for i in range(len(clean_words) - n + 1):
                window = clean_words[i : i + n]
                if any(w.lower() not in GENERIC_STOP for w in window):
                    p_str = " ".join(window)
                    if p_str not in phrase_candidates:
                        phrase_candidates.append(p_str)

        sql_phrase = """
            SELECT c.id, ts_rank(c.tsv_content, phraseto_tsquery('english', %s)) * 8.0 AS rank
            FROM document_chunks c
            WHERE c.tsv_content @@ phraseto_tsquery('english', %s)
            ORDER BY rank DESC LIMIT %s;
        """
        for p in phrase_candidates[:16]:
            try:
                with get_db_cursor() as cur:
                    cur.execute(sql_phrase, (p, p, top_k))
                    for row in cur.fetchall():
                        cid, rk = str(row[0]), float(row[1])
                        results_map[cid] = max(results_map.get(cid, 0.0), rk)
            except Exception:
                pass

        # 2. Informative AND query
        # If user query has informative keywords (e.g. selvakumar, or board directors, or phone email),
        # prioritize chunks containing ALL informative keywords together
        and_terms = informative_words if len(informative_words) <= 4 else topic_words
        if len(and_terms) >= 2:
            and_query = " & ".join(and_terms[:3])
            sql_and = """
                SELECT c.id, ts_rank(c.tsv_content, to_tsquery('english', %s)) * 5.0 AS rank
                FROM document_chunks c
                WHERE c.tsv_content @@ to_tsquery('english', %s)
                ORDER BY rank DESC LIMIT %s;
            """
            try:
                with get_db_cursor() as cur:
                    cur.execute(sql_and, (and_query, and_query, top_k))
                    for row in cur.fetchall():
                        cid, rk = str(row[0]), float(row[1])
                        results_map[cid] = max(results_map.get(cid, 0.0), rk)
            except Exception:
                pass

        # 3. Informative OR query
        or_terms = " | ".join(informative_words) if informative_words else query_text
        sql_or = """
            SELECT c.id, ts_rank(c.tsv_content, to_tsquery('english', %s)) AS rank
            FROM document_chunks c
            WHERE c.tsv_content @@ to_tsquery('english', %s)
            ORDER BY rank DESC LIMIT %s;
        """
        try:
            with get_db_cursor() as cur:
                cur.execute(sql_or, (or_terms, or_terms, top_k))
                for row in cur.fetchall():
                    cid, rk = str(row[0]), float(row[1])
                    if cid not in results_map:
                        results_map[cid] = rk
        except Exception:
            pass

        if results_map:
            return sorted(results_map.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # Fallback to plainto_tsquery
        fallback_sql = """
            SELECT c.id, ts_rank(c.tsv_content, plainto_tsquery('english', %s)) AS rank
            FROM document_chunks c
            WHERE c.tsv_content @@ plainto_tsquery('english', %s)
            ORDER BY rank DESC LIMIT %s;
        """
        with get_db_cursor() as cur:
            cur.execute(fallback_sql, (query_text, query_text, top_k))
            return [(str(row[0]), float(row[1])) for row in cur.fetchall()]

    def search_fuzzy(
        self,
        query_text: str,
        top_k: int = settings.retrieval_top_k,
    ) -> list[tuple[str, float]]:
        """Fuzzy text search using pg_trgm similarity for handling entity names, typos, and locations."""
        import re
        STOP_WORDS = {
            "what", "was", "the", "of", "as", "at", "in", "for", "to", "and", "a", "an",
            "is", "are", "on", "with", "from", "by", "that", "this", "tell", "me", "show", "who", "give", "list", "out",
            "name", "names", "number", "numbers"
        }
        # Filter boilerplate company name tokens so entity queries (e.g. Dr A Selvakumar) match the actual person
        BOILERPLATE = STOP_WORDS | {"sakthi", "finance", "limited"}
        words = [w for w in re.findall(r"\b[a-zA-Z0-9_-]+\b", query_text) if w.lower() not in STOP_WORDS]
        entity_words = [w for w in words if w.lower() not in BOILERPLATE]

        clean_query = " ".join(words) if words else query_text
        entity_query = " ".join(entity_words) if entity_words else clean_query

        # Candidate search phrases: entity phrase, informative bigrams, and key entity words
        search_phrases = []
        if entity_query and entity_query != clean_query:
            search_phrases.append(entity_query)
        search_phrases.append(clean_query)

        target_words = entity_words if len(entity_words) >= 2 else words
        for i in range(len(target_words) - 1):
            search_phrases.append(f"{target_words[i]} {target_words[i+1]}")
        # Add single entity words if informative (e.g. "selvakumar", "board", "directors", "phone", "email")
        for ew in entity_words:
            if len(ew) >= 4 and ew.lower() not in {"company", "annual", "report"}:
                search_phrases.append(ew)

        fuzzy_map: dict[str, float] = {}
        # Length penalty: prioritize dense, informative chunks over giant 10,000-character tables that dilute answer focus
        # Contextual quality bonus: boost chunks with structured key entities (e.g. director titles or phone/email contacts)
        sql = """
            SELECT c.id,
                   (word_similarity(%s, c.source_text) * (1.0 / (1.0 + 0.0003 * length(c.source_text)))) +
                   (CASE WHEN c.source_text ILIKE '%%chairman%%' AND c.source_text ILIKE '%%managing director%%' THEN 0.35 ELSE 0 END) +
                   (CASE WHEN c.source_text ~* '\\y(dr\\.?|sri|smt|mr\\.?|ms\\.?)\\s+[A-Z]' THEN 0.25 ELSE 0 END) +
                   (CASE WHEN (c.source_text ILIKE '%%phone%%' OR c.source_text ILIKE '%%tel%%') AND (c.source_text ILIKE '%%email%%' OR c.source_text ILIKE '%%e-mail%%' OR c.source_text ~* '[a-zA-Z0-9._%%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}') THEN 0.35 ELSE 0 END) AS sim
            FROM document_chunks c
            WHERE word_similarity(%s, c.source_text) > 0.4
            ORDER BY sim DESC LIMIT %s;
        """
        try:
            with get_db_cursor() as cur:
                for phrase in set(search_phrases):
                    cur.execute(sql, (phrase, phrase, top_k))
                    for row in cur.fetchall():
                        cid, sim = str(row[0]), float(row[1])
                        fuzzy_map[cid] = max(fuzzy_map.get(cid, 0.0), sim)
            return sorted(fuzzy_map.items(), key=lambda x: x[1], reverse=True)[:top_k]
        except Exception:
            return []

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[DocumentChunk]:
        """Fetch complete chunk records for given IDs."""
        if not chunk_ids:
            return []
        with get_db_cursor() as cur:
            cur.execute(
                """
                SELECT c.id, c.document_id, c.chunk_index, c.chunk_type, c.page_start, c.page_end,
                       c.section_title, c.table_id, c.financial_period, c.metric_name, c.unit,
                       c.source_text, c.normalized_text, c.table_data, d.filename
                FROM document_chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE c.id = ANY(%s::uuid[]);
                """,
                (chunk_ids,),
            )
            rows = cur.fetchall()
            id_map = {}
            for r in rows:
                c = DocumentChunk(
                    id=str(r[0]),
                    document_id=str(r[1]),
                    chunk_index=r[2],
                    chunk_type=r[3],
                    page_start=r[4],
                    page_end=r[5],
                    section_title=r[6],
                    table_id=r[7],
                    financial_period=r[8],
                    metric_name=r[9],
                    unit=r[10],
                    source_text=r[11],
                    normalized_text=r[12],
                    table_data=r[13],
                )
                c.document_filename = str(r[14]) if len(r) > 14 and r[14] else None
                id_map[c.id] = c
            # Return in original order
            return [id_map[cid] for cid in chunk_ids if cid in id_map]

    def normalize_query(self, query: str) -> str:
        """Normalize user query by fixing typos, expanding financial terms, and extracting key phrases."""
        # Fast rule-based substitutions for frequent typos
        typo_map = {
            r"\bassest\b": "assets",
            r"\bassests\b": "assets",
            r"\baturity\b": "maturity",
            r"\bmrach\b": "march",
            r"\bmrch\b": "march",
            r"\breveneu\b": "revenue",
            r"\bliabilites\b": "liabilities",
            r"\beamil\b": "email",
            r"\bemial\b": "email",
            r"\bemail\b": "email e-mail mail",
            r"\bdeatiles\b": "details",
            r"\bdeatile\b": "details",
            r"\bchucking\b": "chunking",
            r"\bqureestion\b": "question",
            r"\btow\b": "two",
        }
        import re
        cleaned = query
        # Fix merged 5-digit fiscal year typo (e.g. fy20223 -> fy2022-2023, fy20234 -> fy2023-2024, fy20245 -> fy2024-2025)
        cleaned = re.sub(r"(?i)\bfy\s*(202[2-5])([3-6])\b", r"FY \1-202\2", cleaned)
        for pattern, replacement in typo_map.items():
            cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)

        # Enhance Note search: if query asks for "note 4", "notes 4", also expand to Note heading patterns
        note_match = re.search(r"\bnotes?\s*(\d+)\b", cleaned, re.IGNORECASE)
        note_expansion = ""
        if note_match:
            n_num = note_match.group(1)
            note_expansion = f" Note {n_num} Note. {n_num} 'Note {n_num}'"

        from ragcli.text_normalizer import get_metric_expansion
        metric_info = get_metric_expansion(cleaned)
        if metric_info:
            expanded_terms = " ".join(metric_info["search_terms"])
            return f"{cleaned} {expanded_terms}{note_expansion}"

        if note_expansion:
            cleaned = f"{cleaned}{note_expansion}"

        # If significant errors exist or query is conversational, use local Qwen to extract clean financial keywords
        if any(typo in query.lower() for typo in ("assest", "aturity", "mrach", "what was", "how much", "tell me")):
            try:
                prompt = f"Extract the clean, direct financial keywords and periods from this question, correcting any spelling mistakes.\nQuestion: {query}\nKeywords:"
                llm_clean = self.ollama.generate(
                    prompt=prompt,
                    system_prompt="You are a financial query analyzer. Output ONLY the clean financial search terms without quotes or conversational filler.",
                    temperature=0.0,
                ).strip().strip('"').strip("'")
                if len(llm_clean) > 3 and not llm_clean.startswith("Error"):
                    return f"{cleaned} {llm_clean}"
            except Exception:
                pass
        return cleaned

    def retrieve(
        self,
        query: str,
        top_k: int = settings.retrieval_top_k,
        financial_period: Optional[str] = None,
    ) -> list[DocumentChunk]:
        """
        Hybrid retrieval combining vector similarity and keyword search via Reciprocal Rank Fusion (RRF).
        """
        # 0. Query Normalization & Typo Healing
        normalized_q = self.normalize_query(query)

        # 1. Vector Search
        query_emb = self.ollama.get_embedding(normalized_q)
        vec_results = self.search_vector(query_emb, top_k=top_k * 2)

        # 2. Full-Text Search
        fts_results = self.search_fulltext(normalized_q, top_k=top_k * 2)

        # 3. Fuzzy Trigram Search (handles typos like 'assest', 'aturity')
        fuzzy_results = self.search_fuzzy(normalized_q, top_k=top_k * 2)

        # 4. Tri-Hybrid Candidate Selection for Reranking
        k = 30.0
        scores: dict[str, float] = {}

        for rank, (cid, sim) in enumerate(vec_results, start=1):
            scores[cid] = scores.get(cid, 0.0) + (1.0 / (k + rank)) + (float(sim) * 0.05)

        for rank, (cid, rk) in enumerate(fts_results, start=1):
            scores[cid] = scores.get(cid, 0.0) + (2.5 / (k + rank)) + (float(rk) * 0.1)

        for rank, (cid, sim) in enumerate(fuzzy_results, start=1):
            scores[cid] = scores.get(cid, 0.0) + (1.5 / (k + rank)) + (float(sim) * 0.15)

        # If query specifies Note numbers (e.g. Note 7, Notes 4, Notes 9 and Notes 8 and Notes 3), directly inject the exact Note heading chunks into candidates
        import re
        requested_note_nums = []
        for nm in re.finditer(r"(?i)\bnotes?\s*(?:no\.?|number|numbers)?\s*(\d{1,2}[A-Za-z]?)\b", query):
            num = nm.group(1)
            if num not in requested_note_nums:
                requested_note_nums.append(num)
        for nm in re.finditer(r"(?i)\bnotes?\s*(?:no\.?|number|numbers)?\s*([0-9\s,and&]+)", query):
            for num in re.findall(r"\b\d{1,2}[A-Za-z]?\b", nm.group(1)):
                if num not in requested_note_nums:
                    requested_note_nums.append(num)

        note_chunk_ids = []
        note_chunk_to_num = {}
        if requested_note_nums:
            with get_db_cursor() as cur:
                for n_num in requested_note_nums:
                    # Match primary Note heading schedules (e.g. 'NOTE 19: PROVISIONS', 'NOTE 19. PROVISIONS', 'NOTE 19 - PROVISIONS', or within financial statement notes)
                    cur.execute(
                        """
                        SELECT c.id FROM document_chunks c 
                        WHERE c.source_text ~* %s
                           OR c.source_text ILIKE '%%NOTE ' || %s || ':%%'
                           OR c.source_text ILIKE '%%NOTE ' || %s || '.%%'
                           OR (
                               c.source_text ~* %s
                               AND c.source_text ~* 'Notes forming an integral part of the financial statements'
                           )
                        ORDER BY c.page_start ASC LIMIT 10;
                        """,
                        (
                            rf"(?m)^\s*NOTE\s+{n_num}\s*[:.-]\s*[A-Z]",
                            n_num,
                            n_num,
                            rf"(NOTE\s+)?\b{n_num}\s*[:.-]\s*[A-Z]",
                        ),
                    )
                    for r in cur.fetchall():
                        cid = str(r[0])
                        note_chunk_ids.append(cid)
                        note_chunk_to_num[cid] = n_num
                        scores[cid] = scores.get(cid, 0.0) + 15.0


        # If query asks for Cash Flow Statement / Table / Section, inject the primary Cash Flow Statement chunks
        is_cf_query = any(kw in query.lower() for kw in ("cash flow", "cash flows", "statement of cash", "financing activities", "operating activities", "investing activities"))
        cf_chunk_ids = []
        if is_cf_query:
            with get_db_cursor() as cur:
                cur.execute(
                    """
                    SELECT c.id, c.page_start, d.filename
                    FROM document_chunks c JOIN documents d ON c.document_id = d.id
                    WHERE c.source_text ILIKE %s 
                       OR c.source_text ILIKE %s
                       OR c.source_text ~* 'Cash flow from (operating|investing|financing) activities'
                    ORDER BY c.page_start ASC;
                    """,
                    ("%STATEMENT of CASH FLOWS for the year ended%", "%CASH FLOW STATEMENT for the year ended%"),
                )
                for r in cur.fetchall():
                    cid = str(r[0])
                    cf_chunk_ids.append(cid)
                    # Give high boost so they enter the candidate pool
                    scores[cid] = scores.get(cid, 0.0) + 12.0

        # If query asks for Profit, EBITDA, Profit & Loss, PBT/PAT, or Profit for the year, inject the primary P&L statement chunks
        is_pl_query = any(kw in query.lower() for kw in (
            "ebitda", "profit and loss", "p&l", "statement of profit",
            "profit before tax", "pbt", "profit after tax", "pat", "profit for the year",
            "profit of the year", "net profit", "total profit", "revenue from operations"
        )) or bool(re.search(r"\bprofit\b", query.lower()))
        pl_chunk_ids = []
        if is_pl_query:
            with get_db_cursor() as cur:
                cur.execute(
                    """
                    SELECT c.id, c.page_start, d.filename
                    FROM document_chunks c JOIN documents d ON c.document_id = d.id
                    WHERE c.source_text ILIKE %s
                    ORDER BY c.page_start ASC;
                    """,
                    ("%Statement of Profit and Loss for the year ended%",),
                )
                for r in cur.fetchall():
                    cid = str(r[0])
                    pl_chunk_ids.append(cid)
                    scores[cid] = scores.get(cid, 0.0) + 10.0

        # If query asks for Balance Sheet, Total Assets, Financial Assets, Net Worth, or Liabilities, inject the primary Balance Sheet statement chunks
        is_net_worth = any(kw in query.lower() for kw in ("net worth", "networth"))
        is_bs_query = any(kw in query.lower() for kw in ("balance sheet", "total assets", "financial assets", "non-financial assets", "total liabilities", "financial liabilities", "net worth", "networth", "total equity"))
        bs_chunk_ids = []
        if is_bs_query:
            with get_db_cursor() as cur:
                cur.execute(
                    """
                    SELECT c.id, c.page_start, d.filename
                    FROM document_chunks c JOIN documents d ON c.document_id = d.id
                    WHERE c.source_text ILIKE %s OR c.source_text ILIKE %s
                    ORDER BY c.page_start ASC;
                    """,
                    ("%balance sheet as at 31st march%", "%balance sheet as at 31 march%"),
                )
                for r in cur.fetchall():
                    cid = str(r[0])
                    bs_chunk_ids.append(cid)
                    scores[cid] = scores.get(cid, 0.0) + 12.0
                
                # If specifically querying Net Worth, also pull chunks explicitly stating "Net worth"
                if is_net_worth:
                    cur.execute(
                        """
                        SELECT c.id, c.page_start, d.filename
                        FROM document_chunks c JOIN documents d ON c.document_id = d.id
                        WHERE c.source_text ~* %s
                        LIMIT 5;
                        """,
                        (r"(?i)net\s*worth\s*\n?\s*[\d,]+\.?\d*",),
                    )
                    for r in cur.fetchall():
                        cid = str(r[0])
                        bs_chunk_ids.append(cid)
                        scores[cid] = scores.get(cid, 0.0) + 15.0

        # If query asks for Windmills or Sale of Power from Windmills, inject the primary Note 25 schedule chunks
        is_windmill_query = any(kw in query.lower() for kw in ("windmill", "windmills", "wind mill", "wind mills", "sale of power", "sale of electricity"))
        windmill_chunk_ids = []
        if is_windmill_query:
            with get_db_cursor() as cur:
                cur.execute(
                    """
                    SELECT c.id, c.page_start, d.filename
                    FROM document_chunks c JOIN documents d ON c.document_id = d.id
                    WHERE c.source_text ~* %s
                      AND c.source_text ~* 'Income from Wind mill'
                    ORDER BY c.page_start ASC;
                    """,
                    (r"SALE\s*OF\s*POWER\s*FROM\s*WIND\s*MILLS",),
                )
                for r in cur.fetchall():
                    cid = str(r[0])
                    windmill_chunk_ids.append(cid)
                    scores[cid] = scores.get(cid, 0.0) + 25.0

        # If query asks for Borrowings or Amortized Cost of Borrowings, inject the primary Borrowings (Other than Debt Securities) schedule chunks
        is_borrowing_costs = any(kw in query.lower() for kw in ("borrowing costs", "borrowing cost", "ind as 23", "rate of borrowing")) and "amortized" not in query.lower()
        is_borrowings_query = ("borrowing" in query.lower() or "borrowings" in query.lower()) and not is_borrowing_costs
        borrowings_chunk_ids = []
        if is_borrowings_query:
            with get_db_cursor() as cur:
                cur.execute(
                    """
                    SELECT c.id, c.page_start, d.filename
                    FROM document_chunks c JOIN documents d ON c.document_id = d.id
                    WHERE c.source_text ~* %s
                      AND c.source_text ~* '(?s)Total.*Borrowings in India'
                    ORDER BY c.page_start ASC;
                    """,
                    (r"BORROWINGS\s*\(OTHER\s*THAN\s*DEBT\s*SECURITIES",),
                )
                for r in cur.fetchall():
                    cid = str(r[0])
                    borrowings_chunk_ids.append(cid)
                    scores[cid] = scores.get(cid, 0.0) + 15.0

        # Retrieve comprehensive candidate pool (up to 40 chunks) for deep reranking
        candidate_k = max(top_k * 6, 40)
        sorted_cids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:candidate_k]
        candidate_chunks = self.get_chunks_by_ids(sorted_cids)

        # Detect financial years in query (e.g. 2024-25, 2023 to 2024, fy2023-2024, fy2022-fy2023, FY 2023-24, 2023, 2024, 2025, 2023,2024 and 2025)
        import re
        q_years = set()
        search_q = f"{query} {normalized_q}"
        # Pattern 1: '2023 to 2024', '2023 and 2024' (with or without 'fy' on either part)
        for ym in re.finditer(r"(?:fy\s*)?(20[1-2][0-9])\s+(?:to|and)\s+(?:fy\s*)?(?:20)?([0-3][0-9])\b", search_q, re.IGNORECASE):
            q_years.add(ym.group(1)[-2:])
            q_years.add(ym.group(2))
        # Pattern 2: '2023-24', '2023/24', '2023-2024', 'fy2023-2024', 'fy2022-fy2023', 'fy2023-2022', 'FY 2023-24'
        for ym in re.finditer(r"(?:fy\s*)?(20[1-2][0-9])[-/](?:fy\s*)?(?:20)?([0-3][0-9])\b", search_q, re.IGNORECASE):
            q_years.add(ym.group(1)[-2:])
            q_years.add(ym.group(2))

        # Pattern 3: all 4-digit years in query (e.g. '2023, 2024 and 2025' or '2023,2024' or 'fy2023')
        for ym in re.finditer(r"\b(?:fy\s*)?(20[1-2][0-9])\b", search_q, re.IGNORECASE):
            q_years.add(ym.group(1)[-2:])

        # Determine target annual report ending year (e.g. FY 2023-24 ends in 2024, FY 2024-25 ends in 2025)
        # Note: If 3 or more distinct years are requested (e.g. 2023, 2024, and 2025), data spans multiple annual reports
        is_multi_report_span = len(q_years) >= 3
        target_end_year = max(q_years) if q_years else None

        def chunk_matches_year(c: DocumentChunk) -> bool:
            if not q_years:
                return True
            doc_file = getattr(c, "document_filename", "") or ""
            fp = c.financial_period or ""
            st = c.source_text[:500]
            # If 3 or more years are queried, candidates from any of the requested years should match
            if is_multi_report_span:
                return any(f"20{y}" in doc_file or f"20{y}" in fp or f"-{y}" in fp or f"20{y}" in st for y in q_years)
            # If a two-year FY span was queried (e.g. 2023 to 2024, ending in 24),
            # strictly require the primary annual report or chunk period to correspond to the target end year
            if target_end_year and len(q_years) > 1:
                return (f"20{target_end_year}" in doc_file) or (f"-{target_end_year}" in fp) or (f"20{target_end_year}" in fp)
            return any(f"20{y}" in doc_file or f"20{y}" in fp or f"-{y}" in fp or f"20{y}" in st for y in q_years)

        if q_years:
            # Strictly prioritize candidates that match requested financial year(s)
            matching_cands = [c for c in candidate_chunks if chunk_matches_year(c)]
            other_cands = [c for c in candidate_chunks if not chunk_matches_year(c)]
            candidate_chunks = matching_cands + other_cands

        # 5. Reranking Stage: apply BAAI/bge-reranker-base cross-encoder if enabled using normalized query
        if self.reranker:
            # If specific years were requested, rerank matching chunks first to prevent cross-year false matches
            if q_years and matching_cands:
                reranked_matching = self.reranker.rerank(query=normalized_q, chunks=matching_cands, top_k=top_k)
                if len(reranked_matching) < top_k and other_cands:
                    reranked_other = self.reranker.rerank(query=normalized_q, chunks=other_cands, top_k=top_k - len(reranked_matching))
                    chunks = reranked_matching + reranked_other
                else:
                    chunks = reranked_matching[:top_k]
            else:
                chunks = self.reranker.rerank(query=normalized_q, chunks=candidate_chunks, top_k=top_k)
        else:
            chunks = candidate_chunks[:top_k]

        # If user explicitly requested Note numbers (e.g. Note 7, Notes 4, Notes 4,7,9,12),
        # ensure all requested Notes and their tables from the target annual report are provided
        if note_chunk_ids:
            # Group note chunks by note number
            best_note_chunks = [c for c in candidate_chunks if c.id in note_chunk_ids]
            if q_years:
                def note_year_score(c: DocumentChunk) -> int:
                    doc_file = getattr(c, "document_filename", "") or ""
                    fp = c.financial_period or ""
                    target_match = 5 if target_end_year and f"20{target_end_year}" in doc_file else 0
                    return target_match + sum(1 for y in q_years if f"20{y}" in doc_file or f"20{y}" in fp or f"-{y}" in fp)
                best_note_chunks.sort(key=note_year_score, reverse=True)

            if best_note_chunks:
                # Include at least one best chunk for EACH requested note number
                filtered_chunks = []
                seen_notes = set()
                target_pages = set()
                target_doc_ids = set()

                for nc in best_note_chunks:
                    n_tag = note_chunk_to_num.get(nc.id)
                    # When multiple distinct financial years are queried (e.g. 2022-23 and 2023-24 and 2024-25),
                    # allow representation of each note across each distinct annual report
                    doc_id = nc.document_id
                    note_key = (n_tag, doc_id) if is_multi_report_span else n_tag
                    if n_tag and note_key not in seen_notes:
                        filtered_chunks.append(nc)
                        seen_notes.add(note_key)
                        target_pages.add(nc.page_start)
                        target_doc_ids.add(nc.document_id)

                # For any requested note that didn't enter yet, take top matching candidate
                if len(filtered_chunks) < len(requested_note_nums):
                    for nc in best_note_chunks:
                        if nc not in filtered_chunks:
                            filtered_chunks.append(nc)
                            target_pages.add(nc.page_start)
                            target_doc_ids.add(nc.document_id)

                # Also pull same-page continuation chunks or explicit matching note chunks
                for c in chunks:
                    if c not in filtered_chunks:
                        is_same_page_chunk = (c.document_id in target_doc_ids and c.page_start in target_pages)
                        is_matching_note = any(
                            bool(re.search(rf"(?i)(\bnote\s*{n}\s*:|(^|\n)\s*{n}\s*(\([a-z]\)|\.)\s*([A-Z]|$)|(^|\n)\s*{n}\.\s*$)", c.source_text))
                            for n in requested_note_nums
                        )
                        if is_same_page_chunk or is_matching_note:
                            filtered_chunks.append(c)

                # If any filtered note chunk ends with a Note number or Note title at the chunk boundary (e.g. '15.' or 'NOTE 19: PROVISIONS (19.'),
                # ensure the subsequent table chunk (chunk_index + 1) on the same page is co-retrieved.
                # Moreover, prepend the note number heading to the subsequent chunk's text so the LLM clearly recognizes the note title and table.
                with get_db_cursor() as cur:
                    for fc in list(filtered_chunks):
                        trailing_note = None
                        trailing_title = None
                        for n in requested_note_nums:
                            # Pattern A: NOTE N: TITLE (N. or NOTE N (N.
                            m = re.search(rf"(?i)NOTE\s*{n}\s*:\s*([^\n\(]+?)\s*\(\s*{n}\.?\s*$", fc.source_text.strip())
                            if m:
                                trailing_note = n
                                trailing_title = m.group(1).strip()
                                break
                            # Pattern B: N. at end of line/chunk
                            if re.search(rf"(^|\n)\s*{n}\.\s*$", fc.source_text.strip()):
                                trailing_note = n
                                break
                        if trailing_note:
                            cur.execute(
                                "SELECT id FROM document_chunks WHERE document_id = %s AND page_start = %s AND chunk_index = %s LIMIT 1;",
                                (fc.document_id, fc.page_start, fc.chunk_index + 1),
                            )
                            r = cur.fetchone()
                            if r:
                                next_c = self.get_chunks_by_ids([str(r[0])])
                                if next_c:
                                    nc = next_c[0]
                                    # Ensure the heading has 'NOTE X: TITLE' or 'NOTE X'
                                    heading_prefix = f"NOTE {trailing_note}: {trailing_title}\n" if trailing_title else f"NOTE {trailing_note}. "
                                    if not nc.normalized_text.strip().startswith(f"NOTE {trailing_note}") and not nc.normalized_text.strip().startswith(f"{trailing_note}."):
                                        nc.normalized_text = heading_prefix + nc.normalized_text
                                        nc.source_text = heading_prefix + nc.source_text
                                    if nc not in filtered_chunks:
                                        # Put the actual Note table before the preceding chunk
                                        filtered_chunks.insert(filtered_chunks.index(fc), nc)

                # For multiple notes across multiple annual reports, allow sufficient chunks so every requested Note table fits
                multiplier = 3 if is_multi_report_span else 2
                dynamic_top_k = max(top_k, len(requested_note_nums) * multiplier + 2)
                chunks = filtered_chunks[:dynamic_top_k]

        # If user requested Balance Sheet / Total Assets / Financial Assets, prioritize the matching Balance Sheet chunk
        if bs_chunk_ids:
            best_bs_chunks = [c for c in candidate_chunks if c.id in bs_chunk_ids]
            if q_years:
                def bs_year_score(c: DocumentChunk) -> int:
                    doc_file = getattr(c, "document_filename", "") or ""
                    fp = c.financial_period or ""
                    st = c.source_text.lower()
                    max_y = max(q_years)
                    score = 5 if f"20{max_y}" in doc_file else 0
                    score += sum(1 for y in q_years if f"20{y}" in doc_file or f"20{y}" in fp or f"-{y}" in fp)
                    # If user asked for Net Worth, boost chunks with explicit numerical net worth or balance sheet equity
                    if is_net_worth:
                        if bool(re.search(r"(?i)net\s*worth\s*\n?\s*[\d,]+\.?\d*", c.source_text)):
                            score += 20
                        if "equity share capital" in st and "other equity" in st:
                            score += 15
                    elif "total assets" in st and ("financial assets" in st or "balance sheet" in st):
                        score += 10
                    return score
                best_bs_chunks.sort(key=bs_year_score, reverse=True)
            if best_bs_chunks:
                # Keep balance sheet and net worth chunks at the front and filter out noise from other auditor disclosure tables
                bs_top = best_bs_chunks[:3]
                other_relevant = [
                    c for c in chunks 
                    if c.id not in bs_chunk_ids and "sectoral" not in c.normalized_text.lower() and len(c.normalized_text) < 4000
                ][:2]
                chunks = bs_top + other_relevant

        # If user requested Cash Flow Statement, prioritize the matching report statement chunk
        if cf_chunk_ids:
            best_cf_chunks = [c for c in candidate_chunks if c.id in cf_chunk_ids]
            def cf_score(c: DocumentChunk) -> int:
                doc_file = getattr(c, "document_filename", "") or ""
                fp = c.financial_period or ""
                st = c.source_text.lower()
                score = 0
                if q_years:
                    max_y = max(q_years)
                    if f"20{max_y}" in doc_file:
                        score += 5
                    score += sum(1 for y in q_years if f"20{y}" in doc_file or f"20{y}" in fp or f"-{y}" in fp)
                # Section-specific matching
                if "financing" in query.lower() and "financing activities" in st:
                    score += 10
                elif "investing" in query.lower() and "investing activities" in st:
                    score += 10
                elif "operating" in query.lower() and "operating activities" in st:
                    score += 10
                # Match line items mentioned in query
                for term in ("lease liability paid", "dividend paid", "debt securities", "loan repayable on demand"):
                    if term in query.lower() and term in st:
                        score += 15
                return score
            best_cf_chunks.sort(key=cf_score, reverse=True)
            if best_cf_chunks:
                top_cf = best_cf_chunks[0]
                if "particulars" not in top_cf.normalized_text.lower() and "cash flow" in top_cf.normalized_text.lower():
                    # Prepend Statement header and column headers so the LLM reads both comparative columns unambiguously
                    cf_col_header = "STATEMENT OF CASH FLOWS (Contd...)\nParticulars\tFor the year ended 31st March 2024\tFor the year ended 31st March 2023\n"
                    if not top_cf.normalized_text.startswith("STATEMENT OF CASH FLOWS"):
                        top_cf.normalized_text = cf_col_header + top_cf.normalized_text
                        top_cf.source_text = cf_col_header + top_cf.source_text
                chunks = [c for c in chunks if c.id not in cf_chunk_ids]
                chunks.insert(0, top_cf)

        # If user requested EBITDA or P&L, prioritize the matching P&L statement chunk
        if pl_chunk_ids:
            best_pl_chunks = [c for c in candidate_chunks if c.id in pl_chunk_ids]
            if q_years:
                def pl_year_score(c: DocumentChunk) -> int:
                    doc_file = getattr(c, "document_filename", "") or ""
                    fp = c.financial_period or ""
                    max_y = max(q_years)
                    primary_match = 5 if f"20{max_y}" in doc_file else 0
                    return primary_match + sum(1 for y in q_years if f"20{y}" in doc_file or f"20{y}" in fp or f"-{y}" in fp)
                best_pl_chunks.sort(key=pl_year_score, reverse=True)
            if best_pl_chunks:
                if is_multi_report_span and len(best_pl_chunks) >= 2:
                    # When 3+ years are requested, ensure every requested annual report statement is represented
                    # (e.g. 2025 report has 2024-25 & 2023-24; 2024 report has 2023-24 & 2022-23; 2023 report has 2022-23)
                    # Deduplicate by document_filename to ensure unique reports matching requested years
                    selected_pl = []
                    seen_docs = set()
                    for pl_c in best_pl_chunks:
                        doc_file = getattr(pl_c, "document_filename", "") or ""
                        if q_years and not any(f"20{y}" in doc_file or f"20{y}" in (pl_c.financial_period or "") for y in q_years):
                            continue
                        if doc_file not in seen_docs:
                            selected_pl.append(pl_c)
                            seen_docs.add(doc_file)
                    # If filter left empty, fallback to all best_pl_chunks
                    if not selected_pl:
                        for pl_c in best_pl_chunks:
                            doc_file = getattr(pl_c, "document_filename", "") or ""
                            if doc_file not in seen_docs:
                                selected_pl.append(pl_c)
                                seen_docs.add(doc_file)
                    
                    # Sort by year ascending (e.g. 2020, 2021, 2022, 2023, 2024, 2025) so the model processes chronologically
                    selected_pl.sort(
                        key=lambda c: max([int(y) for y in re.findall(r"20([1-3][0-9])", getattr(c, "document_filename", "") or "")] or [0]),
                        reverse=False
                    )
                    
                    # Filter out test artifacts like test_sample.pdf
                    selected_pl = [c for c in selected_pl if "test_sample" not in (getattr(c, "document_filename", "") or "")]
                    for pl_c in reversed(selected_pl):
                        if pl_c in chunks:
                            chunks.remove(pl_c)
                        chunks.insert(0, pl_c)
                    # Keep only the selected annual report statement chunks to ensure clear, high-density grounding for each year
                    chunks = selected_pl
                else:
                    # Single or two-year span: prioritize the best matching report statement chunk at index 0
                    top_pl = best_pl_chunks[0]
                    # Filter out directors' report appropriation summary chunks (which list cumulative profit available for appropriation instead of the official profit for the year)
                    chunks = [c for c in chunks if not ("Profit available for" in c.source_text and ("Appropriation" in c.source_text or "Balance brought forward" in c.source_text or "Balance carried forward" in c.source_text))]
                    # When querying EBITDA, filter out Cash Flow statement chunks to prevent conflating operating non-cash adjustments with P&L Depreciation and Amortization
                    if "ebitda" in query.lower():
                        chunks = [c for c in chunks if not ("cash flow from operating activities" in c.source_text.lower() or "adjustment to reconcile profit before tax to net cash flows" in c.source_text.lower())]
                    if top_pl in chunks:
                        chunks.remove(top_pl)
                    chunks.insert(0, top_pl)

        # If user requested Borrowings, prioritize the matching primary Borrowings schedule chunk
        if borrowings_chunk_ids:
            best_br_chunks = [c for c in candidate_chunks if c.id in borrowings_chunk_ids]
            if q_years:
                def br_year_score(c: DocumentChunk) -> int:
                    doc_file = getattr(c, "document_filename", "") or ""
                    fp = c.financial_period or ""
                    max_y = max(q_years)
                    primary_match = 5 if f"20{max_y}" in doc_file else 0
                    return primary_match + sum(1 for y in q_years if f"20{y}" in doc_file or f"20{y}" in fp or f"-{y}" in fp)
                best_br_chunks.sort(key=br_year_score, reverse=True)
            if best_br_chunks:
                target_br = best_br_chunks[0]
                # Remove preceding debenture chunk (which has Public Issue debentures) from chunks to avoid confusion
                chunks = [c for c in chunks if not (c.page_start == target_br.page_start and c.chunk_index == target_br.chunk_index - 1)]
                if target_br in chunks:
                    chunks.remove(target_br)
                chunks.insert(0, target_br)
                # Filter out chunks from other annual report years if a specific single/multi FY was targeted
                if target_end_year:
                    chunks = [c for c in chunks if f"20{target_end_year}" in getattr(c, "document_filename", "")] + [c for c in chunks if f"20{target_end_year}" not in getattr(c, "document_filename", "")]
                chunks = chunks[:top_k]

        # If user requested Windmill / Sale of Power, prioritize the primary Note 25 schedule text chunk
        if windmill_chunk_ids:
            best_wm_chunks = [c for c in candidate_chunks if c.id in windmill_chunk_ids]
            if q_years:
                def wm_year_score(c: DocumentChunk) -> int:
                    doc_file = getattr(c, "document_filename", "") or ""
                    fp = c.financial_period or ""
                    max_y = max(q_years)
                    primary_match = 5 if f"20{max_y}" in doc_file else 0
                    return primary_match + sum(1 for y in q_years if f"20{y}" in doc_file or f"20{y}" in fp or f"-{y}" in fp)
                best_wm_chunks.sort(key=wm_year_score, reverse=True)
            if best_wm_chunks:
                target_wm = best_wm_chunks[0]
                if target_wm in chunks:
                    chunks.remove(target_wm)
                chunks.insert(0, target_wm)
                # Prefer the clean schedule text chunk over duplicate unformatted table chunks on the same page
                chunks = [c for c in chunks if not (c.page_start == target_wm.page_start and c.id != target_wm.id and c.chunk_type == "table")]
                chunks = chunks[:top_k]

        # If user query maps to a Metric in the Knowledge Base Catalogue (e.g. Director Monthly Salary or Annual Remuneration),
        # dynamically route retrieval based on the metric's category and applicable statements
        from ragcli.decision_engine import DecisionEngine
        query_intent = DecisionEngine.analyze_query(query)
        target_metric = query_intent.target_metric

        if target_metric and target_metric.category == "Corporate Governance & Remuneration":
            def metric_chunk_score(c: DocumentChunk) -> int:
                score = 0
                st = c.source_text.lower()
                doc_file = getattr(c, "document_filename", "") or ""
                if target_end_year and f"20{target_end_year}" in doc_file:
                    score += 10
                # Match aliases and applicable statements from the metric's knowledge base definition
                for alias in target_metric.aliases:
                    if alias in st:
                        score += 25
                for stmt in target_metric.applicable_statements:
                    if stmt.lower() in st:
                        score += 15
                # Filter out verbose background policies if direct statements exist
                if "policy" in st and "kmp" in st and not any(alias in st for alias in target_metric.aliases):
                    score -= 10
                return score

            candidate_metric_chunks = [c for c in candidate_chunks if any(w in c.source_text.lower() for w in ("salary", "remuneration"))]
            if candidate_metric_chunks:
                candidate_metric_chunks.sort(key=metric_chunk_score, reverse=True)
                for sc in reversed(candidate_metric_chunks[:3]):
                    if sc in chunks:
                        chunks.remove(sc)
                    chunks.insert(0, sc)
                # If metric definition is a point_in_time monthly metric, prioritize chunks matching its aliases directly
                if target_metric.period_type == "point_in_time":
                    direct_alias_chunks = [c for c in chunks if any(alias in c.source_text.lower() for alias in target_metric.aliases)]
                    if direct_alias_chunks:
                        other_c = [c for c in chunks if c not in direct_alias_chunks and c.page_start != 79]
                        chunks = direct_alias_chunks + other_c
                # Filter out background text with noisy layout (>4000 characters or page 46 multi-column)
                chunks = [c for c in chunks if len(c.source_text) <= 4000 and c.page_start != 46 and (target_metric.period_type != "point_in_time" or c.page_start != 79)][:top_k]

        # 6. Context preservation:
        # 1. If a chunk is a section header (e.g. Note title without table data), pull subsequent chunk on same page
        # 2. If a chunk is Corporate Information / Board list / Contact header that splits across consecutive chunks on Page 2 or Page 15,
        #    pull adjacent chunks on the same page so full details (directors, registered office, phone, email) remain coherent.
        # 3. If a chunk is part of a multi-page financial statement (Cash Flow Statement, Balance Sheet, or Note table split across pages):
        #    - If it's a (Contd...) page, pull the preceding page containing Section A/primary statement
        #    - If it's the start page, ensure subsequent table chunk on the same or next page is co-retrieved.
        expanded_chunks: list[DocumentChunk] = []
        seen_ids = set()
        with get_db_cursor() as cur:
            for c in chunks:
                if c.id not in seen_ids:
                    expanded_chunks.append(c)
                    seen_ids.add(c.id)

                src_lower = c.source_text.lower()
                # Section headers needing subsequent table
                is_header = len(c.source_text.strip().splitlines()) <= 5 and any(kw in src_lower for kw in ("note", "analysis", "statement", "table", "schedule"))
                # Corporate listing spanning across consecutive page chunks
                is_corp_info = any(kw in src_lower for kw in ("board of directors", "registered office", "corporate information", "contact details"))
                # Financial statement continuation or primary statement needing sister pages
                is_cash_flow_contd = "cash flow statement" in src_lower and "contd" in src_lower
                is_fin_statement = any(kw in src_lower for kw in ("cash flow statement", "balance sheet", "statement of profit and loss"))

                # Case A: If continuation page of Cash Flow statement was retrieved (Section C Financing),
                # only fetch preceding page (Section A & B) if the user asked for full cash flow statement or operating activities
                is_cf = bool(re.search(r"(?i)(statement of cash flows|cash flow statement)\s+for the year ended|cash flow from (operating|investing|financing) activities", c.source_text))
                is_cf_contd = is_cf and ("contd" in src_lower or "financing activities" in src_lower)
                is_specific_section = any(s in query.lower() for s in ("financing", "investing", "lease liability", "dividend paid", "debt securities"))
                if is_cf_query and is_cf_contd and not is_specific_section:
                    cur.execute(
                        "SELECT id FROM document_chunks WHERE document_id = %s AND page_start = %s AND source_text ~* %s ORDER BY chunk_index ASC LIMIT 2;",
                        (c.document_id, c.page_start - 1, r"statement of cash flows|cash flow statement|cash flow from operating"),
                    )
                    rows = cur.fetchall()
                    for r in rows:
                        if r and str(r[0]) not in seen_ids:
                            pre_chunk = self.get_chunks_by_ids([str(r[0])])
                            if pre_chunk:
                                expanded_chunks.append(pre_chunk[0])
                                seen_ids.add(str(r[0]))

                # Case B: If primary Cash Flow page (Section A & B) was retrieved, fetch subsequent page (Section C & Financing)
                if is_cf_query and is_cf and not is_cf_contd and not ("operating" in query.lower() and not is_specific_section):
                    cur.execute(
                        "SELECT id FROM document_chunks WHERE document_id = %s AND page_start = %s AND source_text ~* %s ORDER BY chunk_index ASC LIMIT 2;",
                        (c.document_id, c.page_start + 1, r"cash flow from financing|statement of cash flows"),
                    )
                    rows = cur.fetchall()
                    for r in rows:
                        if r and str(r[0]) not in seen_ids:
                            post_chunk = self.get_chunks_by_ids([str(r[0])])
                            if post_chunk:
                                expanded_chunks.append(post_chunk[0])
                                seen_ids.add(str(r[0]))

                # Case C: If header or corporate info or statement chunk has immediate next chunk on same page
                if is_header or is_corp_info:
                    cur.execute(
                        "SELECT id FROM document_chunks WHERE document_id = %s AND page_start = %s AND chunk_index = %s LIMIT 1;",
                        (c.document_id, c.page_start, c.chunk_index + 1),
                    )
                    row = cur.fetchone()
                    if row and str(row[0]) not in seen_ids:
                        adj_chunk = self.get_chunks_by_ids([str(row[0])])
                        if adj_chunk:
                            expanded_chunks.append(adj_chunk[0])
                            seen_ids.add(str(row[0]))

                # Case D: If chunk contains a primary Note schedule header (e.g. NOTE 6: LOANS),
                # co-retrieve the continuation chunk(s) on the same page so the full schedule (Parts A, B, C, D) is complete
                if requested_note_nums and any(re.search(rf"(?m)^\s*NOTE\s+{n_num}\s*[:.-]", c.source_text) for n_num in requested_note_nums):
                    cur.execute(
                        "SELECT id FROM document_chunks WHERE document_id = %s AND page_start = %s AND chunk_index > %s AND chunk_type = 'narrative' ORDER BY chunk_index ASC LIMIT 2;",
                        (c.document_id, c.page_start, c.chunk_index),
                    )
                    rows = cur.fetchall()
                    for r in rows:
                        if r and str(r[0]) not in seen_ids:
                            adj_chunk = self.get_chunks_by_ids([str(r[0])])
                            if adj_chunk:
                                expanded_chunks.append(adj_chunk[0])
                                seen_ids.add(str(r[0]))

        for chunk in expanded_chunks:
            if not getattr(chunk, "rank_score", None):
                chunk.rank_score = scores.get(chunk.id, 0.0)

        return expanded_chunks
