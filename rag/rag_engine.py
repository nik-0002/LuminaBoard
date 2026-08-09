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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
    - Multi-strategy retrieval (semantic + structured + aggregate)
    - Query routing to relevant datasets
    - Aggregate statistics injection for numeric queries
    - Cross-dataset join hints
    - Document deduplication and diversity
    """

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 3),
            max_features=80_000,
            sublinear_tf=True,
            min_df=1,
            stop_words="english",
        )
        self.documents: List[Dict] = []
        self.tfidf_matrix = None
        self.is_built = False
        self._csv_metadata: Dict[str, Dict] = {}
        self._dataframes: Dict[str, pd.DataFrame] = {}
        self._aggregate_cache: Dict[str, Any] = {}

    # ─── Index Building ────────────────────────────────────────────────────────

    def build_index(self):
        """Load all CSVs and build TF-IDF index + aggregate stats cache."""
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

        # Build TF-IDF
        texts = [d["text"] for d in self.documents]
        logger.info(f"[RAG] Fitting TF-IDF on {len(texts)} docs from {len(csv_paths)} CSVs")
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.is_built = True
        logger.info(f"[RAG] Index ready: {len(self.documents)} docs, shape {self.tfidf_matrix.shape}")

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

            for idx, row in rows_to_index.iterrows():
                parts = [f"Dataset:{source}"]
                for col, val in row.items():
                    sv = str(val)
                    if sv not in ("unknown", "nan", "NaT", ""):
                        parts.append(f"{col.replace('_', ' ')}:{sv}")
                text = " | ".join(parts)
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

        if not self.documents or self.tfidf_matrix is None:
            return []

        results = []

        # ── Strategy 1: TF-IDF Semantic ─────────────────────────────────────
        query_vec = self.vectorizer.transform([query_text])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        if csv_filter:
            filter_name = csv_filter.replace(".csv", "")
            mask = np.array([1.0 if doc["source"] == filter_name else 0.0
                             for doc in self.documents])
            scores = scores * mask

        top_indices = scores.argsort()[::-1][:top_k * 4]
        seen = {}
        for idx in top_indices:
            score = float(scores[idx])
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
        """Extract named entities (state, crop, product) and do exact-match lookup."""
        results = []
        q_lower = query_text.lower()

        # Indian states
        STATES = [
            "andhra pradesh", "telangana", "maharashtra", "karnataka",
            "tamil Nadu", "punjab", "haryana", "uttar pradesh", "madhya pradesh",
            "rajasthan", "gujarat", "west bengal", "odisha", "bihar",
            "jharkhand", "chhattisgarh", "assam", "kerala",
        ]
        mentioned_states = [s for s in STATES if s.lower() in q_lower]

        CROPS = ["rice", "wheat", "cotton", "soybean", "maize", "sugarcane",
                 "tomato", "onion", "chilli", "groundnut", "sunflower", "mustard"]
        mentioned_crops = [c for c in CROPS if c in q_lower]

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