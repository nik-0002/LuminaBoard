"""
Lumina Board - Enhanced CSV RAG Engine v2
Multi-strategy retrieval: TF-IDF semantic search + structured column filters +
aggregate statistics injection. Supports per-dataset weighting and query routing.
"""

import os
import glob
import logging
import json
import re
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime

import pandas as pd
import numpy as np
from rank_bm25 import BM25Okapi

logger = logging.getLogger("lumina.rag")


# Column → semantic topic mapping for smarter query routing
COLUMN_TOPIC_MAP = {
    "grower": ["grower_id", "state", "district", "language", "device_type",
               "grower_age", "gender", "grower_farm_size", "product_scan",
               "offline_campaign_attended"],
    "campaign": ["campaign_id", "social_post_impression", "landing_page_visits",
                 "lead_form_submission", "campaign_crop", "campaign_product",
                 "week_start_date"],
    "retailer": ["retailer_id", "territory_id", "state", "district", "tehsil",
                 "sku_name", "sku_qty", "sku_price", "transaction_date"],
    "inventory": ["sku_id", "sku_name", "sku_qty", "week_end_date", "retailer_id"],
    "rep": ["rep_id", "territory_id", "territory_name", "tehsil_list"],
    "whatsapp": ["campaign_product", "campaign_crop", "grower_id",
                 "delivered_status", "opened_status", "clicked_status"],
    "visit": ["rep_id", "visit_date", "visit_type", "product_recommended"],
}

# Stat functions to run on numeric columns and inject into context
STAT_FUNCTIONS = {
    "sum": np.sum,
    "mean": np.mean,
    "median": np.median,
    "max": np.max,
    "min": np.min,
    "std": np.std,
    "count": len,
}


class CSVRagEngine:
    """
    Enhanced CSV RAG Engine with:
    - BM25 Search Algorithm
    - Contextual Row Serialization
    - Dynamic Entity Extraction (Auto-gazetteer)
    - Query routing to relevant datasets
    - Aggregate statistics injection for numeric queries
    """

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.documents: List[Dict] = []
        self.bm25: Optional[BM25Okapi] = None
        self.is_built = False
        self._csv_metadata: Dict[str, Dict] = {}
        self._dataframes: Dict[str, pd.DataFrame] = {}
        self._aggregate_cache: Dict[str, Any] = {}
        
        # Dynamic entities extracted during indexing
        self._dynamic_states = set()
        self._dynamic_crops = set()
        self._dynamic_products = set()

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenizer that lowercases and removes basic stopwords for BM25."""
        stop_words = {"a", "an", "the", "and", "or", "but", "if", "is", "are", "was", "were", "be", "been", "in", "on", "at", "to", "for", "with", "about", "who", "what", "where", "when", "why", "how", "of", "this", "that", "it", "they", "them"}
        tokens = text.lower().replace(".", "").replace(",", "").replace("?", "").split()
        return [t for t in tokens if t not in stop_words and len(t) > 1]

    # ─── Index Building ────────────────────────────────────────────────────────

    def build_index(self):
        """Load all CSVs and build BM25 index + aggregate stats cache."""
        logger.info(f"[RAG] Building enhanced index from {self.data_dir}")
        self.documents = []
        self._dataframes = {}
        self._aggregate_cache = {}

        csv_paths = self._discover_csvs()
        if not csv_paths:
            logger.warning(f"[RAG] No CSV files found in {self.data_dir}")
            self.is_built = True
            return

        for path in csv_paths:
            self._index_csv(path)

        if not self.documents:
            logger.warning("[RAG] No documents to index.")
            self.is_built = True
            return

        # Build aggregate stats cache for all datasets
        self._build_aggregate_cache()

        # Build BM25 index
        texts = [d["text"] for d in self.documents]
        logger.info(f"[RAG] Tokenizing and fitting BM25 on {len(texts)} docs from {len(csv_paths)} CSVs")
        tokenized_corpus = [self._tokenize(text) for text in texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.is_built = True
        logger.info(f"[RAG] Index ready: {len(self.documents)} docs built with BM25")

    def _discover_csvs(self) -> List[str]:
        paths = set()
        for pattern in ["*.csv", "**/*.csv"]:
            for p in glob.glob(os.path.join(self.data_dir, pattern), recursive=True):
                paths.add(os.path.abspath(p))
        return sorted(paths)

    def _index_csv(self, path: str):
        """Convert CSV rows → searchable text docs with rich metadata."""
        try:
            df = pd.read_csv(path, low_memory=False)
            source = os.path.basename(path).replace(".csv", "")

            # Handle date columns
            for col in df.columns:
                if any(kw in col.lower() for kw in ["date", "datetime", "week"]):
                    try:
                        df[col] = pd.to_datetime(df[col], errors="coerce")
                    except Exception:
                        pass

            df_filled = df.fillna("unknown")
            self._dataframes[source] = df

            self._csv_metadata[source] = {
                "path": path,
                "columns": list(df.columns),
                "row_count": len(df),
                "numeric_cols": df.select_dtypes(include=[np.number]).columns.tolist(),
                "categorical_cols": df.select_dtypes(include=["object"]).columns.tolist(),
            }

            # Build header doc (schema-level)
            header_text = (
                f"Dataset: {source} | Columns: {', '.join(df.columns)} | "
                f"Rows: {len(df)} | Schema overview for {source}"
            )
            self.documents.append({
                "text": header_text,
                "source": source,
                "row_idx": -1,
                "doc_type": "schema",
                "data": {"columns": list(df.columns), "row_count": len(df)},
            })

            # Build row docs (sample: all rows up to 5000, then sample)
            rows_to_index = df_filled
            if len(df_filled) > 5000:
                rows_to_index = df_filled.sample(5000, random_state=42)
                # Always include first 100 rows
                rows_to_index = pd.concat([df_filled.head(100), rows_to_index]).drop_duplicates()

            # Dynamic extraction logic and contextual serialization
            has_state = "state" in df.columns
            has_crop = "crop" in df.columns or "campaign_crop" in df.columns
            has_product = "product" in df.columns or "campaign_product" in df.columns

            for idx, row in rows_to_index.iterrows():
                parts = []
                
                # Dynamic Entity Extraction
                if has_state and str(row.get("state", "")) not in ("unknown", "nan", ""):
                    self._dynamic_states.add(str(row["state"]).lower().strip())
                if "crop" in row and str(row.get("crop", "")) not in ("unknown", "nan", ""):
                    self._dynamic_crops.add(str(row["crop"]).lower().strip())
                if "campaign_crop" in row and str(row.get("campaign_crop", "")) not in ("unknown", "nan", ""):
                    self._dynamic_crops.add(str(row["campaign_crop"]).lower().strip())
                if "product" in row and str(row.get("product", "")) not in ("unknown", "nan", ""):
                    self._dynamic_products.add(str(row["product"]).lower().strip())
                if "campaign_product" in row and str(row.get("campaign_product", "")) not in ("unknown", "nan", ""):
                    self._dynamic_products.add(str(row["campaign_product"]).lower().strip())

                # Contextual Serialization
                context_prefix = f"This record is from the {source} dataset. "
                
                if source == "growers":
                    parts.append(f"{context_prefix}Grower {row.get('grower_id', '')} is in {row.get('district', '')} district, {row.get('state', '')}.")
                    crop_info = str(row.get('grower_crop_calendar', ''))
                    actual_crop = 'various crops'
                    if crop_info and crop_info not in ('unknown', 'nan', 'NaT'):
                        try:
                            crop_data = json.loads(crop_info)
                            actual_crop = crop_data.get('crop', 'various crops')
                        except:
                            actual_crop = 'various crops'
                    parts.append(f"They cultivate {actual_crop}.")
                    parts.append(f"They use a {row.get('device_type', '')} device.")
                elif source == "whatsapp_campaign":
                    parts.append(f"{context_prefix}Campaign {row.get('campaign_id', '')} targeted grower {row.get('grower_id', '')}.")
                    parts.append(f"The campaign is about {row.get('campaign_product', '')} for {row.get('campaign_crop', '')}.")
                elif source == "retailer_pos":
                    parts.append(f"{context_prefix}Retailer {row.get('retailer_id', '')} sold {row.get('sku_qty', 0)} units of {row.get('sku_name', '')}.")
                else:
                    parts.append(context_prefix)
                    
                # Append remaining details
                exclude_cols = ["grower_id", "state", "district", "crop", "device_type", "campaign_id", "campaign_product", "campaign_crop", "sku_name", "retailer_id", "grower_crop_calendar"]
                for col, val in row.items():
                    sv = str(val)
                    if sv not in ("unknown", "nan", "NaT", "") and col not in exclude_cols:
                        parts.append(f"{col.replace('_', ' ')} is {sv}.")
                
                text = " ".join(parts)
                self.documents.append({
                    "text": text,
                    "source": source,
                    "row_idx": int(idx),
                    "doc_type": "row",
                    "data": row.to_dict(),
                })

            logger.info(f"[RAG] Indexed {len(rows_to_index)} rows from {source}")

        except Exception as e:
            logger.error(f"[RAG] Failed to index {path}: {e}")

    def _build_aggregate_cache(self):
        """Pre-compute aggregate stats for fast numeric queries."""
        for source, df in self._dataframes.items():
            cache = {"source": source, "row_count": len(df)}
            numeric_cols = df.select_dtypes(include=[np.number]).columns

            for col in numeric_cols:
                vals = df[col].dropna()
                if len(vals) == 0:
                    continue
                cache[col] = {
                    "sum": float(vals.sum()),
                    "mean": float(vals.mean()),
                    "median": float(vals.median()),
                    "max": float(vals.max()),
                    "min": float(vals.min()),
                    "std": float(vals.std()),
                    "count": int(len(vals)),
                    "pct_nonzero": float((vals != 0).mean() * 100),
                }

            # Categorical distributions (top 10 per col)
            cat_cols = df.select_dtypes(include=["object"]).columns
            for col in cat_cols[:8]:
                vc = df[col].value_counts().head(10)
                cache[f"{col}_dist"] = vc.to_dict()

            self._aggregate_cache[source] = cache

    # ─── Querying ──────────────────────────────────────────────────────────────

    def query(
        self,
        query_text: str,
        top_k: int = 15,
        csv_filter: Optional[str] = None,
        min_score: float = 0.005,
        include_aggregates: bool = True,
    ) -> List[Dict]:
        """
        Multi-strategy retrieval:
        1. TF-IDF semantic similarity
        2. Structured filter injection (state/crop/product mentions)
        3. Aggregate statistics for numeric questions
        """
        if not self.is_built:
            self.build_index()

        if not self.documents or self.bm25 is None:
            return []

        results = []

        # ── Strategy 1: BM25 Lexical/Semantic Search ─────────────────────────────────────
        tokenized_query = self._tokenize(query_text)
        scores = self.bm25.get_scores(tokenized_query)

        if csv_filter:
            filter_name = csv_filter.replace(".csv", "")
            mask = np.array([1.0 if doc["source"] == filter_name else 0.0
                             for doc in self.documents])
            scores = scores * mask

        top_indices = scores.argsort()[::-1][:top_k * 4]
        seen = {}
        for idx in top_indices:
            score = float(scores[idx])
            # BM25 scores are unnormalized and can be > 1.0, so the threshold might need tuning.
            # But we can keep min_score for now to discard absolute zero matches.
            if score < min_score:
                continue
            doc = self.documents[idx]
            key = (doc["source"], doc["row_idx"])
            if key not in seen or seen[key]["score"] < score:
                seen[key] = {**doc, "score": score}

        semantic_results = sorted(seen.values(), key=lambda x: x["score"], reverse=True)[:top_k]
        results.extend(semantic_results)

        # ── Strategy 2: Structured entity extraction ─────────────────────────
        structured = self._structured_query(query_text, csv_filter)
        results.extend(structured)

        # ── Strategy 3: Aggregate stats injection ────────────────────────────
        if include_aggregates:
            agg_docs = self._aggregate_query(query_text, csv_filter)
            results.extend(agg_docs)

        # Deduplicate, sort by score descending
        deduped = {}
        for r in results:
            key = (r["source"], r.get("row_idx", -999))
            if key not in deduped or deduped[key].get("score", 0) < r.get("score", 0):
                deduped[key] = r

        final = sorted(deduped.values(), key=lambda x: x.get("score", 0), reverse=True)
        return final[:top_k]

    def _structured_query(self, query_text: str, csv_filter: Optional[str]) -> List[Dict]:
        """Extract named entities dynamically (state, crop, product) and do exact-match lookup."""
        results = []
        q_lower = query_text.lower()

        # Dynamic entity matching
        mentioned_states = [s for s in self._dynamic_states if s in q_lower]
        mentioned_crops = [c for c in self._dynamic_crops if c in q_lower]
        mentioned_products = [p for p in self._dynamic_products if p in q_lower]

        for source, df in self._dataframes.items():
            if csv_filter and source != csv_filter.replace(".csv", ""):
                continue

            # State filter
            if "state" in df.columns and mentioned_states:
                for state in mentioned_states[:2]:
                    state_df = df[df["state"].str.lower() == state.lower()]
                    if len(state_df) > 0:
                        sample = state_df.head(5)
                        for _, row in sample.iterrows():
                            results.append({
                                "text": f"[STATE_FILTER:{state}] " + " | ".join(
                                    f"{k}:{v}" for k, v in row.items()
                                    if str(v) not in ("nan", "unknown", "")
                                ),
                                "source": source,
                                "row_idx": int(row.name),
                                "doc_type": "structured_filter",
                                "data": row.to_dict(),
                                "score": 0.6,
                            })

            # Crop filter
            for crop_col in ["campaign_crop", "grower_crop_calendar", "crop"]:
                if crop_col in df.columns and mentioned_crops:
                    for crop in mentioned_crops[:2]:
                        crop_df = df[df[crop_col].astype(str).str.lower().str.contains(crop, na=False)]
                        if len(crop_df) > 0:
                            sample = crop_df.head(3)
                            for _, row in sample.iterrows():
                                results.append({
                                    "text": f"[CROP_FILTER:{crop}] " + " | ".join(
                                        f"{k}:{v}" for k, v in row.items()
                                        if str(v) not in ("nan", "unknown", "")
                                    ),
                                    "source": source,
                                    "row_idx": int(row.name),
                                    "doc_type": "structured_filter",
                                    "data": row.to_dict(),
                                    "score": 0.55,
                                })
                            break

            # Product filter
            for prod_col in ["campaign_product", "product", "sku_name"]:
                if prod_col in df.columns and mentioned_products:
                    for product in mentioned_products[:2]:
                        prod_df = df[df[prod_col].astype(str).str.lower().str.contains(product, na=False)]
                        if len(prod_df) > 0:
                            sample = prod_df.head(3)
                            for _, row in sample.iterrows():
                                results.append({
                                    "text": f"[PRODUCT_FILTER:{product}] " + " | ".join(
                                        f"{k}:{v}" for k, v in row.items()
                                        if str(v) not in ("nan", "unknown", "")
                                    ),
                                    "source": source,
                                    "row_idx": int(row.name),
                                    "doc_type": "structured_filter",
                                    "data": row.to_dict(),
                                    "score": 0.55,
                                })
                            break

        return results[:10]

    def _aggregate_query(self, query_text: str, csv_filter: Optional[str]) -> List[Dict]:
        """Inject aggregate statistics relevant to the query."""
        results = []
        q_lower = query_text.lower()

        numeric_keywords = ["total", "sum", "average", "avg", "count", "how many",
                            "percentage", "rate", "max", "highest", "lowest", "trend"]
        if not any(kw in q_lower for kw in numeric_keywords):
            # Still inject a summary doc
            pass

        for source, cache in self._aggregate_cache.items():
            if csv_filter and source != csv_filter.replace(".csv", ""):
                continue

            # Build aggregate summary text
            parts = [f"AGGREGATE STATS for {source} ({cache['row_count']} rows)"]
            for col, stats in cache.items():
                if isinstance(stats, dict) and "sum" in stats:
                    parts.append(
                        f"{col}: sum={stats['sum']:.1f}, mean={stats['mean']:.2f}, "
                        f"max={stats['max']:.1f}, nonzero={stats['pct_nonzero']:.1f}%"
                    )
                elif isinstance(stats, dict) and col.endswith("_dist"):
                    col_name = col.replace("_dist", "")
                    top = list(stats.items())[:5]
                    parts.append(f"{col_name} distribution: " +
                                 ", ".join(f"{k}={v}" for k, v in top))

            agg_text = " | ".join(parts[:20])
            results.append({
                "text": agg_text,
                "source": source,
                "row_idx": -2,
                "doc_type": "aggregate",
                "data": cache,
                "score": 0.3,
            })

        return results

    def query_by_filters(
        self,
        filters: Dict[str, str],
        csv_name: Optional[str] = None,
        limit: int = 50,
    ) -> pd.DataFrame:
        """Direct structured query: filter CSV rows by column=value pairs."""
        results = []
        for source, df in self._dataframes.items():
            if csv_name and source != csv_name:
                continue
            try:
                mask = pd.Series([True] * len(df))
                for col, val in filters.items():
                    if col in df.columns:
                        mask &= df[col].astype(str).str.lower() == str(val).lower()
                matched = df[mask].head(limit)
                if len(matched) > 0:
                    matched = matched.copy()
                    matched["_source"] = source
                    results.append(matched)
            except Exception as e:
                logger.error(f"[RAG] Filter error on {source}: {e}")

        return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

    def get_dataset_summary(self, source: str) -> Dict:
        """Return a rich summary of a specific dataset."""
        if source not in self._dataframes:
            return {}
        df = self._dataframes[source]
        meta = self._csv_metadata.get(source, {})
        agg = self._aggregate_cache.get(source, {})

        summary = {
            "source": source,
            "row_count": len(df),
            "columns": meta.get("columns", []),
            "numeric_cols": meta.get("numeric_cols", []),
            "categorical_cols": meta.get("categorical_cols", []),
            "aggregates": {k: v for k, v in agg.items()
                          if isinstance(v, dict) and "sum" in v},
            "distributions": {k.replace("_dist", ""): v
                              for k, v in agg.items() if k.endswith("_dist")},
            "sample_rows": df.head(5).fillna("").to_dict(orient="records"),
        }
        return summary

    def get_cross_dataset_insights(self) -> Dict:
        """Generate cross-dataset join insights (growers ↔ campaigns ↔ retailers)."""
        insights = {}
        ds = self._dataframes

        # Grower-Campaign linkage
        if "growers" in ds and "whatsapp_campaign" in ds:
            grower_ids = set(ds["growers"]["grower_id"].dropna().unique())
            wa_ids = set(ds["whatsapp_campaign"]["grower_id"].dropna().unique())
            overlap = grower_ids & wa_ids
            insights["grower_whatsapp_coverage"] = {
                "total_growers": len(grower_ids),
                "growers_in_whatsapp": len(overlap),
                "coverage_pct": round(len(overlap) / max(len(grower_ids), 1) * 100, 2),
            }

        # Retailer-Inventory-POS linkage
        if "retailers" in ds and "retailer_pos" in ds:
            ret_ids = set(ds["retailers"]["retailer_id"].dropna().unique())
            pos_ids = set(ds["retailer_pos"]["retailer_id"].dropna().unique())
            overlap = ret_ids & pos_ids
            insights["retailer_pos_coverage"] = {
                "total_retailers": len(ret_ids),
                "retailers_with_pos": len(overlap),
                "coverage_pct": round(len(overlap) / max(len(ret_ids), 1) * 100, 2),
            }

        return insights

    def get_status(self) -> Dict:
        return {
            "is_built": self.is_built,
            "total_documents": len(self.documents),
            "csv_files": list(self._csv_metadata.keys()),
            "csv_metadata": self._csv_metadata,
            "matrix_shape": list(self.tfidf_matrix.shape) if self.tfidf_matrix is not None else None,
            "aggregate_cache_keys": list(self._aggregate_cache.keys()),
            "cross_dataset_insights": self.get_cross_dataset_insights() if self._dataframes else {},
        }