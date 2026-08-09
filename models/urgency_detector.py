"""
Lumina Board - Enhanced Bio-Urgency Detector v2
Detects agricultural bio-emergencies using:
- ML anomaly detection (IsolationForest) on campaign + sales data
- Rule-based seasonal crop risk scoring
- Inventory depletion signals (protective pesticides running low)
- Grower engagement churn detection
- WhatsApp campaign delivery failure patterns
- Regional POS velocity drops (precursor to stockout)
- Weather-proxy signals from data patterns
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("lumina.urgency")


# ─── Bio-Emergency Product Mapping ────────────────────────────────────────────
# Products that are specifically protective — their depletion = urgency spike
PROTECTIVE_PRODUCTS = {
    "fungicide": ["Amistar", "Ridomil", "Score", "Cabrio", "Tilt", "Mancozeb"],
    "insecticide": ["Karate", "Actara", "Vertimec", "Proclaim", "Ampligo"],
    "herbicide": ["Topik", "Axial", "Dual Gold", "Gramoxone"],
    "crop_protection": ["Amistar Top", "Revus", "Quadris"],
}

# Crop-specific pest/disease risk windows (month-based)
CROP_BIO_RISK = {
    "rice": {
        "months": [7, 8, 9],
        "threats": ["Blast disease", "Brown plant hopper", "Stem borer"],
        "severity": 85,
        "protective_products": ["Amistar", "Ridomil", "Karate"],
    },
    "cotton": {
        "months": [8, 9, 10],
        "threats": ["Bollworm", "Whitefly", "Pink bollworm"],
        "severity": 90,
        "protective_products": ["Ampligo", "Actara", "Karate"],
    },
    "wheat": {
        "months": [2, 3],
        "threats": ["Yellow rust", "Karnal bunt", "Aphids"],
        "severity": 75,
        "protective_products": ["Tilt", "Amistar Top", "Topik"],
    },
    "soybean": {
        "months": [8, 9],
        "threats": ["Soybean mosaic virus", "Root rot", "Girdle beetle"],
        "severity": 70,
        "protective_products": ["Ridomil", "Score", "Actara"],
    },
    "maize": {
        "months": [7, 8],
        "threats": ["Fall armyworm", "Stalk rot", "Leaf blight"],
        "severity": 80,
        "protective_products": ["Ampligo", "Tilt", "Karate"],
    },
    "chilli": {
        "months": [9, 10, 11],
        "threats": ["Thrips", "Mites", "Leaf curl virus"],
        "severity": 75,
        "protective_products": ["Vertimec", "Actara", "Karate"],
    },
}

# State-level historical pest-risk multipliers (index > 1.0 = elevated risk)
STATE_RISK_MULTIPLIER = {
    "Andhra Pradesh": 1.35,
    "Telangana": 1.30,
    "Maharashtra": 1.20,
    "Karnataka": 1.15,
    "Tamil Nadu": 1.10,
    "Punjab": 1.05,
    "Haryana": 1.00,
    "Uttar Pradesh": 1.15,
    "Madhya Pradesh": 1.10,
    "Rajasthan": 0.95,
    "Gujarat": 1.00,
    "West Bengal": 1.25,
    "Odisha": 1.20,
}


class UrgencyDetector:
    """
    Comprehensive bio-urgency detection across 6 signal dimensions:
    1. Campaign anomaly (ML + rule-based)
    2. Inventory depletion of protective products
    3. Seasonal crop-specific bio-risk
    4. Grower engagement churn
    5. POS sales velocity change
    6. WhatsApp delivery failure rate
    """

    URGENCY_LEVELS = {
        (0, 25): ("LOW", "🟢", "No immediate action needed"),
        (25, 50): ("MEDIUM", "🟡", "Monitor closely — some signals present"),
        (50, 75): ("HIGH", "🟠", "Action recommended within 1 week"),
        (75, 101): ("CRITICAL", "🔴", "Immediate intervention required"),
    }

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self._isolation_forest: Optional[IsolationForest] = None
        self._scaler = StandardScaler()
        self._is_trained = False
        self._trained_on_rows = 0

    # ─── Training ─────────────────────────────────────────────────────────────

    def train(self, datasets: Optional[Dict[str, pd.DataFrame]] = None):
        logger.info("[Urgency] Training detector...")
        if datasets is None:
            try:
                from utils.data_processors import DataProcessor
                dp = DataProcessor(self.data_dir)
                datasets = dp.load_all_datasets()
            except Exception as e:
                logger.warning(f"[Urgency] Dataset load failed: {e}")
                self._is_trained = True
                return

        features = self._build_training_features(datasets)
        if features is not None and len(features) > 5:
            try:
                scaled = self._scaler.fit_transform(features)
                self._isolation_forest = IsolationForest(
                    n_estimators=150,
                    contamination=0.08,
                    random_state=42,
                    max_samples="auto",
                )
                self._isolation_forest.fit(scaled)
                self._trained_on_rows = len(features)
                logger.info(f"[Urgency] IsolationForest trained on {len(features)} vectors")
            except Exception as e:
                logger.error(f"[Urgency] Training failed: {e}")

        self._is_trained = True

    def _build_training_features(self, datasets: Dict) -> Optional[np.ndarray]:
        """Build multi-source feature matrix for anomaly detection."""
        feature_rows = []

        if "campaigns" in datasets:
            c = datasets["campaigns"]
            for _, row in c.iterrows():
                impr = max(row.get("social_post_impression", 0), 1)
                visits = row.get("landing_page_visits", 0)
                leads = row.get("lead_form_submission", 0)
                ctr = visits / impr * 100
                conv = leads / max(visits, 1) * 100
                feature_rows.append([ctr, conv, np.log1p(impr), np.log1p(visits)])

        if "retailer_pos" in datasets:
            pos = datasets["retailer_pos"]
            if "sku_qty" in pos.columns and "sku_price" in pos.columns:
                pos = pos.copy()
                pos["revenue"] = pos["sku_qty"] * pos["sku_price"]
                # Weekly aggregation proxy
                for _, grp in pos.groupby("retailer_id"):
                    rev = float(grp["revenue"].sum())
                    qty = float(grp["sku_qty"].sum())
                    feature_rows.append([qty / max(len(grp), 1), rev / max(len(grp), 1), 0, 0])

        if not feature_rows:
            return None

        arr = np.array(feature_rows, dtype=float)
        return np.nan_to_num(arr, nan=0.0)

    # ─── Main Detection ────────────────────────────────────────────────────────

    def detect(
        self,
        datasets: Dict[str, pd.DataFrame],
        state_filter: Optional[str] = None,
        crop_filter: Optional[str] = None,
        product_filter: Optional[str] = None,
    ) -> Dict:
        if not self._is_trained:
            self.train(datasets)

        signals: List[Dict] = []
        score_components: Dict[str, float] = {}

        # ── 1. Campaign Anomaly ──────────────────────────────────────────────
        if "campaigns" in datasets:
            s, sc = self._signal_campaign(datasets["campaigns"], state_filter, crop_filter, product_filter)
            signals.extend(s)
            score_components["campaign_anomaly"] = sc

        # ── 2. Inventory Depletion ───────────────────────────────────────────
        if "retailer_inventory" in datasets:
            s, sc = self._signal_inventory(datasets["retailer_inventory"], state_filter, product_filter)
            signals.extend(s)
            score_components["inventory_depletion"] = sc

        # ── 3. Bio-Seasonal Risk ─────────────────────────────────────────────
        s, sc = self._signal_seasonal_bio(crop_filter, state_filter)
        signals.extend(s)
        score_components["bio_seasonal"] = sc

        # ── 4. Grower Churn ──────────────────────────────────────────────────
        if "growers" in datasets:
            s, sc = self._signal_grower_churn(datasets["growers"], state_filter)
            signals.extend(s)
            score_components["grower_churn"] = sc

        # ── 5. POS Velocity ──────────────────────────────────────────────────
        if "retailer_pos" in datasets:
            s, sc = self._signal_pos_velocity(datasets["retailer_pos"], state_filter, product_filter)
            signals.extend(s)
            score_components["pos_velocity"] = sc

        # ── 6. WhatsApp Delivery ─────────────────────────────────────────────
        if "whatsapp" in datasets:
            s, sc = self._signal_whatsapp(datasets["whatsapp"], state_filter)
            signals.extend(s)
            score_components["whatsapp_delivery"] = sc

        # ── 7. ML Anomaly Overlay ────────────────────────────────────────────
        if self._isolation_forest is not None and "campaigns" in datasets:
            ml_sc = self._ml_anomaly_score(datasets["campaigns"], state_filter, crop_filter)
            score_components["ml_anomaly"] = ml_sc

        # ── Composite with state multiplier ─────────────────────────────────
        weights = {
            "campaign_anomaly": 0.30,
            "inventory_depletion": 0.25,
            "bio_seasonal": 0.20,
            "grower_churn": 0.10,
            "pos_velocity": 0.08,
            "whatsapp_delivery": 0.04,
            "ml_anomaly": 0.03,
        }
        composite = sum(score_components.get(k, 0) * w for k, w in weights.items())
        multiplier = STATE_RISK_MULTIPLIER.get(state_filter, 1.0) if state_filter else 1.0
        composite = min(100, composite * multiplier)

        level_info = self._get_level(composite)
        recommendations = self._generate_recommendations(signals, composite, state_filter, crop_filter, datasets)

        # Sort signals by severity desc
        signals.sort(key=lambda x: x.get("severity", 0), reverse=True)

        return {
            "urgency_score": round(composite, 1),
            "urgency_level": level_info[0],
            "urgency_icon": level_info[1],
            "urgency_message": level_info[2],
            "signals": signals[:8],
            "score_breakdown": {k: round(v, 1) for k, v in score_components.items()},
            "recommendations": recommendations,
            "affected_crops": self._identify_affected_crops(crop_filter, signals),
            "suggested_products": self._suggest_products(crop_filter, state_filter),
            "filters_applied": {
                "state": state_filter,
                "crop": crop_filter,
                "product": product_filter,
            },
            "analyzed_at": datetime.utcnow().isoformat(),
        }

    # ─── Signal Generators ────────────────────────────────────────────────────

    def _signal_campaign(
        self,
        df: pd.DataFrame,
        state: Optional[str],
        crop: Optional[str],
        product: Optional[str],
    ) -> Tuple[List[Dict], float]:
        signals = []
        score = 0.0

        mask = pd.Series([True] * len(df))
        if crop and "campaign_crop" in df.columns:
            mask &= df["campaign_crop"].str.lower().str.contains(crop.lower(), na=False)
        if product and "campaign_product" in df.columns:
            mask &= df["campaign_product"].str.lower().str.contains(product.lower(), na=False)

        filtered = df[mask]
        if len(filtered) == 0:
            return signals, 0.0

        global_ctr = (
            df["landing_page_visits"].sum() / max(df["social_post_impression"].sum(), 1) * 100
            if "landing_page_visits" in df.columns and "social_post_impression" in df.columns else 3.0
        )
        global_conv = (
            df["lead_form_submission"].sum() / max(df["landing_page_visits"].sum(), 1) * 100
            if "lead_form_submission" in df.columns and "landing_page_visits" in df.columns else 5.0
        )

        if "social_post_impression" in filtered.columns and "landing_page_visits" in filtered.columns:
            total_impr = filtered["social_post_impression"].sum()
            total_visits = filtered["landing_page_visits"].sum()
            ctr = total_visits / max(total_impr, 1) * 100

            if ctr < global_ctr * 0.4:
                signals.append({
                    "type": "campaign_ctr_critical",
                    "message": f"CTR is {ctr:.2f}% — critically below platform avg {global_ctr:.2f}%",
                    "severity": 85,
                    "data": {"ctr": round(ctr, 2), "avg_ctr": round(global_ctr, 2)},
                })
                score = 85
            elif ctr < global_ctr * 0.65:
                signals.append({
                    "type": "campaign_ctr_low",
                    "message": f"CTR {ctr:.2f}% is below platform avg {global_ctr:.2f}%",
                    "severity": 50,
                    "data": {"ctr": round(ctr, 2), "avg_ctr": round(global_ctr, 2)},
                })
                score = max(score, 50)

        if "landing_page_visits" in filtered.columns and "lead_form_submission" in filtered.columns:
            visits = filtered["landing_page_visits"].sum()
            subs = filtered["lead_form_submission"].sum()
            conv = subs / max(visits, 1) * 100

            if conv < global_conv * 0.3:
                signals.append({
                    "type": "conversion_critical",
                    "message": f"Lead conversion {conv:.2f}% — severely below avg {global_conv:.2f}%",
                    "severity": 80,
                    "data": {"conversion_pct": round(conv, 2), "global_conv": round(global_conv, 2)},
                })
                score = max(score, 80)

        # Zero-submission check
        if "lead_form_submission" in filtered.columns:
            zero_pct = (filtered["lead_form_submission"] == 0).mean() * 100
            if zero_pct > 40:
                signals.append({
                    "type": "zero_submission_bulk",
                    "message": f"{zero_pct:.0f}% of campaign rows have zero lead submissions",
                    "severity": 65,
                    "data": {"zero_pct": round(zero_pct, 1)},
                })
                score = max(score, 65)

        return signals, min(score, 100)

    def _signal_inventory(
        self,
        df: pd.DataFrame,
        state: Optional[str],
        product: Optional[str],
    ) -> Tuple[List[Dict], float]:
        signals = []
        score = 0.0

        if "sku_qty" not in df.columns or "sku_name" not in df.columns:
            return signals, 0.0

        # Filter to protective products
        prot_df = df[df["sku_name"].str.lower().apply(
            lambda name: any(
                p.lower() in name
                for group in PROTECTIVE_PRODUCTS.values()
                for p in group
            )
        )]

        if len(prot_df) == 0:
            return signals, 0.0

        # Identify zero/low stock
        out_of_stock = prot_df[prot_df["sku_qty"] == 0]
        low_stock = prot_df[(prot_df["sku_qty"] > 0) & (prot_df["sku_qty"] < 5)]

        if len(prot_df) > 0:
            oos_pct = len(out_of_stock) / len(prot_df) * 100
            if oos_pct > 30:
                signals.append({
                    "type": "protective_stockout_critical",
                    "message": f"{oos_pct:.0f}% of protective product SKUs are out of stock",
                    "severity": 90,
                    "data": {
                        "oos_count": int(len(out_of_stock)),
                        "total_protective_sku_records": int(len(prot_df)),
                        "oos_pct": round(oos_pct, 1),
                        "oos_products": out_of_stock["sku_name"].value_counts().head(5).to_dict(),
                    },
                })
                score = 90
            elif oos_pct > 15:
                signals.append({
                    "type": "protective_stockout_high",
                    "message": f"{oos_pct:.0f}% protective product SKUs out of stock — reorder urgently",
                    "severity": 65,
                    "data": {"oos_pct": round(oos_pct, 1)},
                })
                score = max(score, 65)

        # Weekly depletion trend (if week_end_date available)
        if "week_end_date" in df.columns:
            try:
                df2 = df.copy()
                df2["week_end_date"] = pd.to_datetime(df2["week_end_date"], errors="coerce")
                weekly = df2.groupby("week_end_date")["sku_qty"].mean().sort_index()
                if len(weekly) >= 3:
                    recent = weekly.iloc[-3:]
                    if recent.iloc[-1] < recent.iloc[0] * 0.7:
                        signals.append({
                            "type": "inventory_depletion_trend",
                            "message": f"Inventory depleting: avg stock fell {((1 - recent.iloc[-1]/max(recent.iloc[0], 0.01)) * 100):.0f}% over 3 weeks",
                            "severity": 55,
                            "data": {"weekly_trend": recent.to_dict()},
                        })
                        score = max(score, 55)
            except Exception:
                pass

        return signals, min(score, 100)

    def _signal_seasonal_bio(
        self,
        crop: Optional[str],
        state: Optional[str],
    ) -> Tuple[List[Dict], float]:
        signals = []
        score = 0.0
        month = datetime.now().month

        if crop:
            crop_lower = crop.lower()
            for crop_key, info in CROP_BIO_RISK.items():
                if crop_key in crop_lower:
                    if month in info["months"]:
                        sev = info["severity"]
                        state_mult = STATE_RISK_MULTIPLIER.get(state, 1.0) if state else 1.0
                        adj_sev = min(100, sev * state_mult)
                        signals.append({
                            "type": "bio_seasonal_peak",
                            "message": (
                                f"PEAK BIO-RISK for {crop}: "
                                f"{', '.join(info['threats'][:2])} — active threat window"
                            ),
                            "severity": adj_sev,
                            "data": {
                                "threats": info["threats"],
                                "protective_products": info["protective_products"],
                                "month": month,
                                "state_risk_mult": state_mult,
                            },
                        })
                        score = adj_sev
                    elif month in [m - 1 for m in info["months"] if m > 1] + [m + 1 for m in info["months"] if m < 12]:
                        signals.append({
                            "type": "bio_seasonal_approaching",
                            "message": f"Bio-risk window approaching for {crop} — prepare {info['protective_products'][0]}",
                            "severity": info["severity"] * 0.55,
                            "data": {"threats": info["threats"], "months": info["months"]},
                        })
                        score = max(score, info["severity"] * 0.55)
                    break

        # Kharif/Rabi general window
        if month in [6, 7, 8]:
            signals.append({
                "type": "kharif_peak",
                "message": "Kharif peak season: high pest pressure across India — maximum agrochemical demand",
                "severity": 60 if not score else 30,
                "data": {"season": "kharif", "month": month},
            })
            score = max(score, 60 if not crop else score)
        elif month in [11, 12]:
            signals.append({
                "type": "rabi_sowing",
                "message": "Rabi sowing season active — wheat/mustard protection window open",
                "severity": 50 if not score else 25,
                "data": {"season": "rabi", "month": month},
            })
            score = max(score, 50 if not crop else score)

        return signals, min(score, 100)

    def _signal_grower_churn(
        self,
        df: pd.DataFrame,
        state: Optional[str],
    ) -> Tuple[List[Dict], float]:
        signals = []
        score = 0.0

        filtered = df[df["state"] == state] if state and "state" in df.columns else df
        if len(filtered) == 0:
            return signals, 0.0

        # Scan rate
        if "product_scan" in filtered.columns:
            scan_rate = filtered["product_scan"].mean() * 100
            if scan_rate < 15:
                signals.append({
                    "type": "low_grower_scan_rate",
                    "message": f"Only {scan_rate:.1f}% growers actively scanning products — engagement at risk",
                    "severity": 55,
                    "data": {"scan_rate_pct": round(scan_rate, 2)},
                })
                score = 55
            elif scan_rate < 30:
                signals.append({
                    "type": "moderate_scan_rate",
                    "message": f"Product scan rate {scan_rate:.1f}% — below healthy 30% threshold",
                    "severity": 35,
                    "data": {"scan_rate_pct": round(scan_rate, 2)},
                })
                score = max(score, 35)

        # Campaign attendance
        if "offline_campaign_attended" in filtered.columns:
            att_rate = filtered["offline_campaign_attended"].mean() * 100
            if att_rate < 10:
                signals.append({
                    "type": "low_offline_engagement",
                    "message": f"Offline campaign attendance only {att_rate:.1f}% — physical reach failing",
                    "severity": 45,
                    "data": {"attendance_rate_pct": round(att_rate, 2)},
                })
                score = max(score, 45)

        # Device type — keypad dominance
        if "device_type" in filtered.columns:
            keypad_pct = (filtered["device_type"] == "keypad").mean() * 100
            if keypad_pct > 55:
                signals.append({
                    "type": "digital_divide",
                    "message": f"{keypad_pct:.0f}% keypad users — digital campaigns ineffective, voice/SMS critical",
                    "severity": 50,
                    "data": {"keypad_pct": round(keypad_pct, 1)},
                })
                score = max(score, 50)

        # Small farm concentration → price sensitivity
        if "grower_farm_size" in filtered.columns:
            small_pct = (filtered["grower_farm_size"] < 2).mean() * 100
            if small_pct > 65:
                signals.append({
                    "type": "small_farm_risk",
                    "message": f"{small_pct:.0f}% farms <2 acres — high price sensitivity, ROI messaging essential",
                    "severity": 35,
                    "data": {"small_farm_pct": round(small_pct, 1)},
                })
                score = max(score, 35)

        return signals, min(score, 100)

    def _signal_pos_velocity(
        self,
        df: pd.DataFrame,
        state: Optional[str],
        product: Optional[str],
    ) -> Tuple[List[Dict], float]:
        signals = []
        score = 0.0

        if "transaction_date" not in df.columns or "sku_qty" not in df.columns:
            return signals, 0.0

        try:
            df2 = df.copy()
            df2["transaction_date"] = pd.to_datetime(df2["transaction_date"], errors="coerce")
            df2 = df2.dropna(subset=["transaction_date"])

            if len(df2) == 0:
                return signals, 0.0

            max_date = df2["transaction_date"].max()
            cutoff = max_date - timedelta(days=60)
            recent = df2[df2["transaction_date"] >= cutoff]
            older = df2[df2["transaction_date"] < cutoff]

            if len(recent) > 0 and len(older) > 0:
                recent_daily = len(recent) / 60
                older_daily = len(older) / max((max_date - df2["transaction_date"].min()).days - 60, 1)
                if older_daily > 0:
                    velocity_change = (recent_daily - older_daily) / older_daily * 100
                    if velocity_change < -30:
                        signals.append({
                            "type": "pos_velocity_drop",
                            "message": f"POS transaction velocity dropped {abs(velocity_change):.0f}% — demand signal weakening",
                            "severity": 65,
                            "data": {
                                "velocity_change_pct": round(velocity_change, 1),
                                "recent_daily_avg": round(recent_daily, 2),
                            },
                        })
                        score = 65
                    elif velocity_change > 50:
                        signals.append({
                            "type": "pos_velocity_surge",
                            "message": f"POS velocity surged +{velocity_change:.0f}% — potential emergency buying",
                            "severity": 40,
                            "data": {"velocity_change_pct": round(velocity_change, 1)},
                        })
                        score = max(score, 40)

        except Exception as e:
            logger.debug(f"[Urgency] POS velocity calc error: {e}")

        return signals, min(score, 100)

    def _signal_whatsapp(
        self,
        df: pd.DataFrame,
        state: Optional[str],
    ) -> Tuple[List[Dict], float]:
        signals = []
        score = 0.0

        if "delivered_status" not in df.columns:
            return signals, 0.0

        total = len(df)
        if total == 0:
            return signals, 0.0

        del_rate = df["delivered_status"].mean() * 100
        open_rate = df["opened_status"].mean() * 100 if "opened_status" in df.columns else 0
        click_rate = df["clicked_status"].mean() * 100 if "clicked_status" in df.columns else 0

        if del_rate < 70:
            signals.append({
                "type": "whatsapp_delivery_low",
                "message": f"WhatsApp delivery rate {del_rate:.1f}% — number quality issues suspected",
                "severity": 60,
                "data": {"delivery_rate": round(del_rate, 2)},
            })
            score = 60
        elif del_rate < 85:
            signals.append({
                "type": "whatsapp_delivery_moderate",
                "message": f"WhatsApp delivery {del_rate:.1f}% — below optimal 85%",
                "severity": 30,
                "data": {"delivery_rate": round(del_rate, 2)},
            })
            score = max(score, 30)

        if open_rate < 20 and del_rate > 80:
            signals.append({
                "type": "whatsapp_open_low",
                "message": f"Open rate only {open_rate:.1f}% despite {del_rate:.1f}% delivery — messaging relevance issue",
                "severity": 40,
                "data": {"open_rate": round(open_rate, 2), "delivery_rate": round(del_rate, 2)},
            })
            score = max(score, 40)

        return signals, min(score, 100)

    def _ml_anomaly_score(
        self,
        df: pd.DataFrame,
        state: Optional[str],
        crop: Optional[str],
    ) -> float:
        if self._isolation_forest is None:
            return 0.0

        mask = pd.Series([True] * len(df))
        if crop and "campaign_crop" in df.columns:
            mask &= df["campaign_crop"].str.lower().str.contains(crop.lower(), na=False)

        filtered = df[mask]
        if len(filtered) == 0:
            return 0.0

        features = self._build_training_features({"campaigns": filtered})
        if features is None or len(features) == 0:
            return 0.0

        try:
            scaled = self._scaler.transform(features)
            scores = self._isolation_forest.decision_function(scaled)
            anomaly_pct = (scores < 0).mean() * 100
            return min(anomaly_pct * 1.8, 100)
        except Exception as e:
            logger.debug(f"[Urgency] ML score error: {e}")
            return 0.0

    # ─── Utilities ─────────────────────────────────────────────────────────────

    def _get_level(self, score: float) -> Tuple[str, str, str]:
        for (lo, hi), info in self.URGENCY_LEVELS.items():
            if lo <= score < hi:
                return info
        return ("LOW", "🟢", "No immediate action needed")

    def _identify_affected_crops(self, crop: Optional[str], signals: List[Dict]) -> List[str]:
        if crop:
            return [crop]
        affected = set()
        month = datetime.now().month
        for crop_key, info in CROP_BIO_RISK.items():
            if month in info["months"]:
                affected.add(crop_key)
        return list(affected)[:5]

    def _suggest_products(self, crop: Optional[str], state: Optional[str]) -> List[str]:
        month = datetime.now().month
        suggestions = set()

        if crop:
            crop_lower = crop.lower()
            for crop_key, info in CROP_BIO_RISK.items():
                if crop_key in crop_lower:
                    suggestions.update(info.get("protective_products", []))
                    break

        # General seasonal suggestions
        if month in [7, 8, 9]:
            suggestions.update(["Amistar", "Karate", "Actara"])
        elif month in [11, 12, 1, 2]:
            suggestions.update(["Tilt", "Topik", "Amistar Top"])

        return list(suggestions)[:6]

    def _generate_recommendations(
        self,
        signals: List[Dict],
        score: float,
        state: Optional[str],
        crop: Optional[str],
        datasets: Dict,
    ) -> List[str]:
        recs = []
        signal_types = {s["type"] for s in signals}
        products = self._suggest_products(crop, state)

        if score >= 75:
            recs.append(f"🚨 CRITICAL: Escalate to regional manager — immediate campaign push needed in {state or 'affected region'}")

        if "bio_seasonal_peak" in signal_types:
            prod = products[0] if products else "fungicide/insecticide"
            recs.append(f"⚡ Deploy emergency SMS campaign for {prod} targeting {crop or 'crop'} growers NOW")
            recs.append(f"📞 Activate IVR voice call blast to all {state or 'regional'} growers within 24 hours")

        if "protective_stockout_critical" in signal_types:
            recs.append(f"📦 Emergency stock replenishment for protective products — coordinate with supply chain immediately")

        if "campaign_ctr_critical" in signal_types:
            recs.append("🎯 Pause underperforming campaigns — A/B test new creatives with crop-disease imagery")

        if "conversion_critical" in signal_types:
            recs.append("📋 Simplify lead capture to name+phone only — current form has too much friction")
            recs.append("📞 Deploy telesales follow-up to high-impression, low-conversion grower segments")

        if "digital_divide" in signal_types:
            recs.append("📱 Shift 60%+ budget to SMS/IVR — smartphone penetration insufficient for digital-only")

        if "pos_velocity_drop" in signal_types:
            recs.append("🏪 Rep visit activation for low-velocity retailers — in-store demonstration needed")

        if "whatsapp_delivery_low" in signal_types:
            recs.append("📱 Audit grower phone number quality — refresh contact database for WhatsApp campaigns")

        if "low_grower_scan_rate" in signal_types:
            recs.append(f"📲 Retailer-led QR scan drives in {state or 'region'} — gamify with lucky draw incentives")

        if products and score >= 40:
            recs.append(f"✦ Priority products for deployment: {', '.join(products[:3])}")

        if state and score >= 50:
            recs.append(f"📍 Mobilize field reps in {state} — territory visits within 48–72 hours")

        return recs[:7]

    # ─── Bulk Scan ─────────────────────────────────────────────────────────────

    def bulk_scan(self, datasets: Dict[str, pd.DataFrame]) -> Dict:
        if not self._is_trained:
            self.train(datasets)

        states = []
        if "growers" in datasets and "state" in datasets["growers"].columns:
            states = datasets["growers"]["state"].dropna().unique().tolist()

        results = []
        for state in states[:25]:
            try:
                detection = self.detect(datasets, state_filter=state)
                results.append({
                    "state": state,
                    "urgency_score": detection["urgency_score"],
                    "urgency_level": detection["urgency_level"],
                    "urgency_icon": detection["urgency_icon"],
                    "top_signal": detection["signals"][0]["message"] if detection["signals"] else "No signals",
                    "signal_count": len(detection["signals"]),
                    "suggested_products": detection.get("suggested_products", [])[:2],
                })
            except Exception as e:
                logger.error(f"[Urgency] Bulk scan error for {state}: {e}")

        results.sort(key=lambda x: x["urgency_score"], reverse=True)
        critical = [r for r in results if r["urgency_level"] in ("CRITICAL", "HIGH")]

        return {
            "total_states_scanned": len(results),
            "critical_states": len(critical),
            "results": results,
            "top_hotspots": results[:5],
            "scanned_at": datetime.utcnow().isoformat(),
        }