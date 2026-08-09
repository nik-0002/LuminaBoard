"""
Lumina Board - Enhanced Agricultural Marketing API with Qwen2.5 Integration
Flask backend integrating CSV-based RAG, local Qwen2.5 LLM, urgency detection,
multilingual campaign generation, MSG91 SMS, WhatsApp, and comprehensive data analytics.
"""

import os
import json
import logging
import glob
import re
import requests
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import traceback

import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS

# Internal modules
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.rag_engine import CSVRagEngine
from models.urgency_detector import UrgencyDetector
from messaging.campaign_generator import CampaignMessageGenerator
from messaging.channels.twillio_gateway import TwilioMessagingGateway
from messaging.channels.msg91_gateway import MSG91Gateway
from messaging.channels.fast2sms_gateway import Fast2SMSGateway
from messaging.channels.adb_whatsapp import WhatsAppChannel
from messaging.orchestrator import MessagingOrchestrator, MessageChannel
from messaging.auto_dispatcher import AutomatedMessageDispatcher
from utils.data_processors import DataProcessor

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.FileHandler("../logs/api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("lumina.api")

# ─── App Setup ────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="../dashboard", static_url_path="")
CORS(app)

class ApiPrefixMiddleware(object):
    """Ensure Flask matches routes whether Vercel passes /api/route or /route"""
    def __init__(self, app):
        self.app = app
    def __call__(self, environ, start_response):
        path = environ.get('PATH_INFO', '')
        if path and not path.startswith('/api'):
            environ['PATH_INFO'] = '/api' + path
        return self.app(environ, start_response)

app.wsgi_app = ApiPrefixMiddleware(app.wsgi_app)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(BASE_DIR, "data"))
QWEN_API_URL = os.environ.get("QWEN_API_URL", "http://localhost:11434/api/generate")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen2.5:latest")

# ─── Singletons (lazy-init) ───────────────────────────────────────────────────
_rag_engine: Optional[CSVRagEngine] = None
_urgency_detector: Optional[UrgencyDetector] = None
_campaign_gen: Optional[CampaignMessageGenerator] = None
_twilio_gateway: Optional[TwilioMessagingGateway] = None
_msg91_gateway: Optional[MSG91Gateway] = None
_fast2sms_gateway: Optional[Fast2SMSGateway] = None
_whatsapp_channel: Optional[WhatsAppChannel] = None
_messaging_orchestrator: Optional[MessagingOrchestrator] = None
_auto_dispatcher: Optional[AutomatedMessageDispatcher] = None
_data_processor: Optional[DataProcessor] = None
_datasets: Dict[str, pd.DataFrame] = {}
_data_cache: Dict[str, Any] = {}
DEFAULT_TEST_PHONE = "8978518496"


def get_rag_engine() -> CSVRagEngine:
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = CSVRagEngine(DATA_DIR)
        _rag_engine.build_index()
    return _rag_engine


def get_urgency_detector() -> UrgencyDetector:
    global _urgency_detector
    if _urgency_detector is None:
        _urgency_detector = UrgencyDetector(DATA_DIR)
        _urgency_detector.train()
    return _urgency_detector


def get_campaign_gen() -> CampaignMessageGenerator:
    global _campaign_gen
    if _campaign_gen is None:
        gemini_key = os.environ.get("GEMINI_API_KEY", "")
        try:
            import yaml
            config_path = os.path.join(BASE_DIR, "config", "api_keys.yaml")
            with open(config_path, 'r') as f:
                cfg = yaml.safe_load(f)
            g_key = cfg.get("gemini", {}).get("api_key", "")
            if g_key and not g_key.startswith("${"):
                gemini_key = g_key
        except Exception:
            pass
        _campaign_gen = CampaignMessageGenerator(gemini_api_key=gemini_key)
        if gemini_key:
            logger.info("Campaign Message Generator initialized with Google Gemini API")
    return _campaign_gen


def get_twilio_gateway() -> Optional[TwilioMessagingGateway]:
    """Initialize Twilio gateway from config"""
    global _twilio_gateway
    if _twilio_gateway is None:
        try:
            import yaml
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "api_keys.yaml")
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            twilio_config = config.get('messaging', {}).get('twilio', {})
            if twilio_config.get('enabled', False):
                account_sid = twilio_config.get('account_sid')
                auth_token = twilio_config.get('auth_token')
                phone_number = twilio_config.get('phone_number')
                
                # Check if using environment variables
                if account_sid and account_sid.startswith('${') and account_sid.endswith('}'):
                    env_var = account_sid[2:-1]
                    account_sid = os.environ.get(env_var)
                if auth_token and auth_token.startswith('${') and auth_token.endswith('}'):
                    env_var = auth_token[2:-1]
                    auth_token = os.environ.get(env_var)
                if phone_number and phone_number.startswith('${') and phone_number.endswith('}'):
                    env_var = phone_number[2:-1]
                    phone_number = os.environ.get(env_var)
                
                if account_sid and auth_token and phone_number:
                    _twilio_gateway = TwilioMessagingGateway(account_sid, auth_token, phone_number)
                    logger.info("Twilio gateway initialized successfully")
                else:
                    logger.warning("Twilio credentials not fully configured")
            else:
                logger.info("Twilio is disabled in config")
        except Exception as e:
            logger.error(f"Failed to initialize Twilio gateway: {e}")
    return _twilio_gateway


def get_msg91_gateway() -> Optional[MSG91Gateway]:
    """Initialize MSG91 gateway from config"""
    global _msg91_gateway
    if _msg91_gateway is None:
        try:
            import yaml
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "api_keys.yaml")
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            msg91_config = config.get('messaging', {}).get('msg91', {})
            if msg91_config.get('enabled', False):
                auth_key = msg91_config.get('auth_key', '')
                sender_id = msg91_config.get('sender_id', 'SYNGTA')
                template_id = msg91_config.get('template_id', '')
                use_simple_api = msg91_config.get('use_simple_api', True)
                simulate = msg91_config.get('simulate', True)
                
                # Check if using environment variables
                if auth_key and auth_key.startswith('${') and auth_key.endswith('}'):
                    env_var = auth_key[2:-1]
                    auth_key = os.environ.get(env_var, '')
                
                # If no real auth key, enable simulation
                if not auth_key or auth_key.startswith('${'):
                    simulate = True
                    auth_key = None
                
                _msg91_gateway = MSG91Gateway(
                    auth_key=auth_key,
                    sender_id=sender_id,
                    template_id=template_id if template_id else None,
                    use_simple_api=use_simple_api,
                    simulate=simulate
                )
                logger.info(f"MSG91 gateway initialized (simulate={simulate})")
            else:
                logger.info("MSG91 is disabled in config")
        except Exception as e:
            logger.error(f"Failed to initialize MSG91 gateway: {e}")
    return _msg91_gateway


def get_whatsapp_channel() -> Optional[WhatsAppChannel]:
    """Initialize WhatsApp channel from config"""
    global _whatsapp_channel
    if _whatsapp_channel is None:
        try:
            import yaml
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "api_keys.yaml")
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            wa_config = config.get('messaging', {}).get('whatsapp', {})
            if wa_config.get('enabled', False):
                mode = wa_config.get('mode', 'simulate')
                device_id = wa_config.get('adb', {}).get('device_id', '')
                access_token = wa_config.get('business_api', {}).get('access_token', '')
                phone_number_id = wa_config.get('business_api', {}).get('phone_number_id', '')
                business_account_id = wa_config.get('business_api', {}).get('business_account_id', '')
                webhook_verify_token = wa_config.get('business_api', {}).get('webhook_verify_token', '')
                
                # Resolve environment variables
                if access_token and access_token.startswith('${') and access_token.endswith('}'):
                    env_var = access_token[2:-1]
                    access_token = os.environ.get(env_var, '')
                if phone_number_id and phone_number_id.startswith('${') and phone_number_id.endswith('}'):
                    env_var = phone_number_id[2:-1]
                    phone_number_id = os.environ.get(env_var, '')
                
                # If business_api mode but no credentials, fall back to simulate
                if mode == 'business_api' and (not access_token or not phone_number_id):
                    logger.warning("WhatsApp Business API credentials not found, falling back to simulation")
                    mode = 'simulate'
                
                _whatsapp_channel = WhatsAppChannel(
                    mode=mode,
                    device_id=device_id if device_id else None,
                    access_token=access_token if access_token else None,
                    phone_number_id=phone_number_id if phone_number_id else None,
                    business_account_id=business_account_id if business_account_id else None,
                    webhook_verify_token=webhook_verify_token if webhook_verify_token else None
                )
                logger.info(f"WhatsApp channel initialized (mode={mode})")
            else:
                logger.info("WhatsApp is disabled in config")
        except Exception as e:
            logger.error(f"Failed to initialize WhatsApp channel: {e}")
    return _whatsapp_channel


def get_fast2sms_gateway() -> Optional[Fast2SMSGateway]:
    """Initialize Fast2SMS gateway from config"""
    global _fast2sms_gateway
    if _fast2sms_gateway is None:
        try:
            import yaml
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "api_keys.yaml")
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            f2s_config = config.get('messaging', {}).get('fast2sms', {})
            if f2s_config.get('enabled', True):
                api_key = f2s_config.get('api_key', '')
                simulate = f2s_config.get('simulate', True)
                
                if api_key and api_key.startswith('${') and api_key.endswith('}'):
                    env_var = api_key[2:-1]
                    api_key = os.environ.get(env_var, '')
                
                if not api_key or api_key.startswith('${'):
                    simulate = True
                    api_key = None
                
                _fast2sms_gateway = Fast2SMSGateway(
                    api_key=api_key,
                    simulate=simulate
                )
                logger.info(f"Fast2SMS gateway initialized (simulate={simulate})")
            else:
                logger.info("Fast2SMS is disabled in config")
        except Exception as e:
            logger.error(f"Failed to initialize Fast2SMS gateway: {e}")
    return _fast2sms_gateway


def get_messaging_orchestrator() -> Optional[MessagingOrchestrator]:
    """Initialize messaging orchestrator with all available channels"""
    global _messaging_orchestrator
    if _messaging_orchestrator is None:
        try:
            # Get Fast2SMS gateway (Free India SMS)
            fast2sms_gateway = get_fast2sms_gateway()
            
            # Get MSG91 gateway
            msg91_gateway = get_msg91_gateway()
            
            # Get WhatsApp channel (ADB + Business API + Simulation)
            whatsapp_channel = get_whatsapp_channel()
            
            # Try to get ADB controller
            adb_controller = None
            try:
                from messaging.channels.adb_whatsapp import WhatsAppChannel as WAC
                if not whatsapp_channel:
                    adb_controller = WAC(mode="adb")
            except:
                pass
            
            # Get Twilio gateway
            twilio_gateway = get_twilio_gateway()
            
            # Determine if we should simulate
            simulate = True
            if fast2sms_gateway and not fast2sms_gateway.simulate:
                simulate = False
            if msg91_gateway and not msg91_gateway.simulate:
                simulate = False
            if whatsapp_channel and whatsapp_channel.mode != 'simulate':
                simulate = False
            
            _messaging_orchestrator = MessagingOrchestrator(
                msg91_gateway=msg91_gateway,
                whatsapp_channel=whatsapp_channel,
                twilio_gateway=twilio_gateway,
                fast2sms_gateway=fast2sms_gateway,
                adb_controller=adb_controller,
                simulate=simulate
            )
            logger.info(f"Messaging orchestrator initialized (simulate={simulate})")
        except Exception as e:
            logger.error(f"Failed to initialize messaging orchestrator: {e}")
    return _messaging_orchestrator


def get_auto_dispatcher() -> Optional[AutomatedMessageDispatcher]:
    """Initialize Automated Message Dispatcher"""
    global _auto_dispatcher
    if _auto_dispatcher is None:
        try:
            orchestrator = get_messaging_orchestrator()
            urgency_det = get_urgency_detector()
            campaign_gen = get_campaign_gen()
            
            _auto_dispatcher = AutomatedMessageDispatcher(
                orchestrator=orchestrator,
                urgency_detector=urgency_det,
                campaign_generator=campaign_gen,
                datasets_provider=get_datasets,
                poll_interval_seconds=300
            )
            logger.info("Automated Message Dispatcher initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Automated Message Dispatcher: {e}")
    return _auto_dispatcher


def get_datasets() -> Dict[str, pd.DataFrame]:
    """Load all CSV datasets with proper type handling"""
    global _datasets, _data_processor
    if not _datasets:
        _data_processor = DataProcessor(DATA_DIR)
        _datasets = {}
        
        # Load all CSVs comprehensively
        csv_files = {
            'growers': 'growers.csv',
            'retailers': 'retailers.csv',
            'retailer_pos': 'retailer_pos.csv',
            'retailer_inventory': 'retailer_inventory_weekly.csv',
            'retailer_visits': 'retailer_visit_log.csv',
            'reps_territory': 'reps_territory.csv',
            'campaigns': 'digital_funnel_weekly.csv',
            'whatsapp': 'whatsapp_campaign.csv'
        }
        
        for key, filename in csv_files.items():
            filepath = os.path.join(DATA_DIR, filename)
            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath, low_memory=False)
                    # Convert date columns
                    date_columns = [col for col in df.columns if 'date' in col.lower() or 'datetime' in col.lower()]
                    for col in date_columns:
                        try:
                            df[col] = pd.to_datetime(df[col], errors='coerce')
                        except:
                            pass
                    _datasets[key] = df
                    logger.info(f"Loaded {key}: {len(df)} rows, {len(df.columns)} columns")
                except Exception as e:
                    logger.error(f"Error loading {filename}: {e}")
    
    return _datasets


def call_qwen_api(prompt: str, system_prompt: str = None, max_tokens: int = 2000) -> str:
    """
    Call AI model: Always use Google Gemini API first for intelligent Q&A and chat.
    """
    # 1. Resolve Google Gemini API Key
    camp_gen = get_campaign_gen()
    gemini_key = (camp_gen.gemini_api_key if camp_gen else None) or os.environ.get("GEMINI_API_KEY")
    
    if not gemini_key or str(gemini_key).startswith("${"):
        try:
            import yaml
            config_path = os.path.join(BASE_DIR, "config", "api_keys.yaml")
            with open(config_path, 'r') as f:
                cfg = yaml.safe_load(f)
            g_k = cfg.get("gemini", {}).get("api_key", "")
            if g_k and not g_k.startswith("${"):
                gemini_key = g_k
        except Exception:
            pass

    if gemini_key and len(str(gemini_key).strip()) > 5:
        full_text = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        payload = {
            "contents": [{"parts": [{"text": full_text}]}],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": max_tokens
            }
        }
        for model in ["gemini-flash-latest", "gemini-2.0-flash", "gemini-pro-latest"]:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={gemini_key.strip()}"
            for attempt in range(2):
                try:
                    resp = requests.post(url, json=payload, timeout=15)
                    if resp.status_code == 200:
                        data = resp.json()
                        text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                        logger.info(f"Generated AI response via Google Gemini API ({model}, {len(text)} chars)")
                        return text
                    elif resp.status_code == 429:
                        logger.warning(f"Gemini API 429 rate limit for {model}, retrying in 1.5s...")
                        import time
                        time.sleep(1.5)
                    else:
                        logger.warning(f"Gemini API ({model}) status {resp.status_code}: {resp.text[:100]}")
                        break
                except Exception as ge:
                    logger.error(f"Gemini API call exception ({model}): {ge}")
                    break

    # 2. Try local Ollama API fallback
    try:
        payload = {
            "model": QWEN_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": max_tokens
            }
        }
        if system_prompt:
            payload["system"] = system_prompt
        
        response = requests.post(QWEN_API_URL, json=payload, timeout=(2, 15))
        response.raise_for_status()
        result = response.json()
        return result.get('response', '').strip()
    
    except Exception as e:
        logger.error(f"Ollama API connection failed: {e}")
        context_preview = ""
        if "CURRENT DATA CONTEXT:" in prompt:
            parts = prompt.split("CURRENT DATA CONTEXT:")
            if len(parts) > 1:
                context_preview = parts[1].split("USER QUERY:")[0].strip()
        
        return (
            "✨ **Lumina Board AI Data Synthesis**:\n\n"
            + (f"**Data Summary retrieved from CSV:**\n```\n{context_preview[:1000]}\n```" if context_preview else "No dataset context found for this query.")
        )


def analyze_dataframe_comprehensive(df: pd.DataFrame, name: str) -> Dict[str, Any]:
    """Comprehensive analysis of a dataframe"""
    analysis = {
        'name': name,
        'shape': {'rows': len(df), 'cols': len(df.columns)},
        'columns': list(df.columns),
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
        'null_counts': df.isnull().sum().to_dict(),
        'null_percentages': (df.isnull().sum() / len(df) * 100).to_dict(),
    }
    
    # Numeric columns statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        analysis['numeric_stats'] = df[numeric_cols].describe().to_dict()
    
    # Categorical columns value counts (top 10)
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    analysis['categorical_stats'] = {}
    for col in categorical_cols[:10]:  # Limit to first 10 to avoid huge output
        value_counts = df[col].value_counts().head(10).to_dict()
        analysis['categorical_stats'][col] = value_counts
    
    # Date range for date columns
    date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    analysis['date_ranges'] = {}
    for col in date_cols:
        valid_dates = df[col].dropna()
        if len(valid_dates) > 0:
            analysis['date_ranges'][col] = {
                'min': str(valid_dates.min()),
                'max': str(valid_dates.max()),
                'range_days': (valid_dates.max() - valid_dates.min()).days
            }
    
    return analysis


# ─── Serve Dashboard ──────────────────────────────────────────────────────────
@app.route("/")
def serve_dashboard():
    return send_from_directory("../dashboard", "index.html")




# ─── Health ───────────────────────────────────────────────────────────────────
@app.route("/api/health", methods=["GET"])
def health():
    ds = get_datasets()
    camp_gen = get_campaign_gen()
    fast2sms = get_fast2sms_gateway()
    return jsonify({
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "datasets_loaded": list(ds.keys()),
        "record_counts": {k: len(v) for k, v in ds.items()},
        "ai_engine": "Google Gemini 1.5 Flash API",
        "gemini_api_configured": bool(camp_gen and camp_gen.gemini_api_key),
        "fast2sms_configured": bool(fast2sms and fast2sms.api_key and not fast2sms.simulate)
    })


# ─── CSV Listing & Comprehensive Analysis ─────────────────────────────────────
@app.route("/api/csv/list", methods=["GET"])
def list_csv_files():
    """List all available CSV files with comprehensive metadata"""
    csv_files = glob.glob(os.path.join(DATA_DIR, "**/*.csv"), recursive=True)
    csv_files += glob.glob(os.path.join(DATA_DIR, "*.csv"))
    csv_files = list(set(csv_files))
    
    result = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, nrows=1)
            full_df = pd.read_csv(f, low_memory=False)
            size = os.path.getsize(f)
            
            result.append({
                "name": os.path.basename(f),
                "path": f,
                "columns": list(df.columns),
                "column_count": len(df.columns),
                "row_count": len(full_df),
                "size_kb": round(size / 1024, 2),
                "size_mb": round(size / 1024 / 1024, 2),
                "dtypes": {col: str(dtype) for col, dtype in full_df.dtypes.items()}
            })
        except Exception as e:
            result.append({"name": os.path.basename(f), "error": str(e)})
    
    return jsonify({"files": result, "count": len(result)})


@app.route("/api/csv/comprehensive-analysis", methods=["GET"])
def comprehensive_csv_analysis():
    """Comprehensive analysis of all CSV datasets"""
    ds = get_datasets()
    
    analyses = {}
    for name, df in ds.items():
        analyses[name] = analyze_dataframe_comprehensive(df, name)
    
    # Cross-dataset insights
    total_records = sum(len(df) for df in ds.values())
    total_columns = sum(len(df.columns) for df in ds.values())
    
    summary = {
        'total_datasets': len(ds),
        'total_records': total_records,
        'total_columns': total_columns,
        'datasets': analyses,
        'timestamp': datetime.utcnow().isoformat()
    }
    
    return jsonify(summary)


@app.route("/api/csv/preview", methods=["GET"])
def preview_csv():
    """Preview rows from a CSV file"""
    filename = request.args.get("file")
    rows = int(request.args.get("rows", 20))
    if not filename:
        return jsonify({"error": "file parameter required"}), 400
    
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404
    
    try:
        df = pd.read_csv(filepath, nrows=rows, low_memory=False)
        df = df.where(pd.notnull(df), None)
        return jsonify({
            "file": filename,
            "columns": list(df.columns),
            "rows": df.to_dict(orient="records"),
            "total_shown": len(df)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/csv/stats", methods=["GET"])
def csv_stats():
    """Return comprehensive statistical summary of a CSV file"""
    filename = request.args.get("file")
    if not filename:
        return jsonify({"error": "file parameter required"}), 400
    
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404
    
    try:
        df = pd.read_csv(filepath, low_memory=False)
        analysis = analyze_dataframe_comprehensive(df, filename)
        return jsonify(analysis)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─── Enhanced RAG Chat with Qwen2.5 ───────────────────────────────────────────
@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Enhanced chat endpoint with local Qwen2.5 integration.
    Uses comprehensive CSV RAG to retrieve ALL relevant data,
    then generates grounded responses via local Qwen2.5 API.
    """
    body = request.get_json(force=True)
    query = body.get("query", "").strip()
    history = body.get("history", [])
    csv_filter = body.get("csv_filter")
    
    if not query:
        return jsonify({"error": "query required"}), 400
    
    try:
        # Get all datasets for comprehensive context
        ds = get_datasets()
        
        # Build comprehensive data context
        context_parts = []
        
        # If specific data query, provide relevant statistics
        if any(keyword in query.lower() for keyword in ['grower', 'farmer', 'campaign', 'retailer', 'sales', 'inventory']):
            for name, df in ds.items():
                if csv_filter and name != csv_filter:
                    continue
                
                relevant = False
                if 'grower' in query.lower() and 'grower' in name:
                    relevant = True
                elif 'campaign' in query.lower() and 'campaign' in name:
                    relevant = True
                elif 'retailer' in query.lower() and 'retailer' in name:
                    relevant = True
                elif 'sales' in query.lower() and 'pos' in name:
                    relevant = True
                elif not csv_filter:
                    relevant = True
                
                if relevant:
                    # Add comprehensive dataset summary
                    summary = f"\n[Dataset: {name}]\n"
                    summary += f"Total Records: {len(df)}\n"
                    summary += f"Columns: {', '.join(df.columns.tolist()[:10])}\n"
                    
                    # Add sample data
                    if len(df) > 0:
                        sample = df.head(5).to_dict(orient='records')
                        summary += f"Sample Data: {json.dumps(sample, default=str)}\n"
                    
                    context_parts.append(summary)
        
        # Also use RAG engine for semantic search
        try:
            rag = get_rag_engine()
            retrieved = rag.query(query, top_k=15, csv_filter=csv_filter)
            
            for item in retrieved:
                context_parts.append(
                    f"[Source: {item['source']} | Row {item['row_idx']}]\n{item['text']}"
                )
        except Exception as rag_error:
            logger.error(f"RAG error: {rag_error}")
        
        context_str = "\n\n".join(context_parts) if context_parts else "No specific data context available."
        
        # Build conversation context
        conv_context = ""
        for msg in history[-5:]:  # Last 5 messages
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            conv_context += f"{role.upper()}: {content}\n"
        
        # System prompt for AI Assistant
        system_prompt = f"""You are Lumina AI — an intelligent agricultural marketing assistant for Lumina Board.
You have comprehensive access to real agricultural data including grower profiles, campaign performance, 
retailer analytics, sales data, and territory information from CSV files.

CRITICAL RULES:
1. ONLY use information from the provided data context. NEVER invent statistics or numbers.
2. If data doesn't contain enough information, explicitly state what's missing.
3. Always cite which dataset and specific metrics you're referencing.
4. Be conversational but precise. Format numbers with proper separators.
5. When showing data, present it clearly with proper formatting.
6. For trends, calculate from actual data in context.
7. Provide actionable insights based on the data.
8. If asked about something not in the data, say so and offer related information you do have.

AVAILABLE DATASETS:
{', '.join(ds.keys())}

CURRENT DATA CONTEXT:
{context_str}

CONVERSATION HISTORY:
{conv_context}"""
        
        # Call Qwen2.5
        full_prompt = f"{system_prompt}\n\nUSER QUERY: {query}\n\nASSISTANT:"
        
        response = call_qwen_api(full_prompt, max_tokens=3000)
        
        # Extract sources used
        sources_used = list(set([item['source'] for item in retrieved])) if 'retrieved' in locals() else []
        
        return jsonify({
            "response": response,
            "sources": sources_used,
            "context_size": len(context_str),
            "datasets_consulted": list(ds.keys()),
            "timestamp": datetime.utcnow().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Chat error: {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


# ─── Enhanced Metrics Endpoints ───────────────────────────────────────────────
@app.route("/api/metrics/overview", methods=["GET"])
def metrics_overview():
    """Comprehensive dashboard overview using ALL data"""
    try:
        ds = get_datasets()
        
        overview = {
            "timestamp": datetime.utcnow().isoformat(),
            "total_growers": len(ds.get('growers', pd.DataFrame())),
            "total_retailers": len(ds.get('retailers', pd.DataFrame())),
            "total_territories": len(ds.get('reps_territory', pd.DataFrame())),
            "active_campaigns": len(ds.get('campaigns', pd.DataFrame())['campaign_id'].unique()) if 'campaigns' in ds else 0,
        }
        
        # Calculate revenue from POS data
        if 'retailer_pos' in ds:
            pos = ds['retailer_pos']
            if 'sku_price' in pos.columns and 'sku_qty' in pos.columns:
                pos['revenue'] = pos['sku_price'] * pos['sku_qty']
                overview['total_revenue'] = float(pos['revenue'].sum())
                overview['avg_transaction_value'] = float(pos['revenue'].mean())
                overview['total_transactions'] = len(pos)
        
        # Campaign performance
        if 'campaigns' in ds:
            camp = ds['campaigns']
            if 'lead_form_submission' in camp.columns:
                overview['total_leads'] = int(camp['lead_form_submission'].sum())
            if 'landing_page_visits' in camp.columns:
                overview['total_visits'] = int(camp['landing_page_visits'].sum())
                overview['overall_conversion_rate'] = round(
                    (overview.get('total_leads', 0) / overview.get('total_visits', 1)) * 100, 2
                )
        
        # Grower engagement
        if 'growers' in ds:
            growers = ds['growers']
            if 'product_scan' in growers.columns:
                overview['growers_engaged'] = int(growers['product_scan'].sum())
                overview['engagement_rate'] = round(
                    (overview['growers_engaged'] / len(growers)) * 100, 2
                )
        
        # Retailer activity
        if 'retailer_visits' in ds:
            visits = ds['retailer_visits']
            overview['total_field_visits'] = len(visits)
            if 'visit_date' in visits.columns:
                visits['visit_date'] = pd.to_datetime(visits['visit_date'], errors='coerce')
                recent_visits = visits[visits['visit_date'] >= datetime.now() - timedelta(days=30)]
                overview['visits_last_30_days'] = len(recent_visits)
        
        # WhatsApp campaign stats
        if 'whatsapp' in ds:
            wa = ds['whatsapp']
            overview['whatsapp_messages_sent'] = len(wa)
            if 'delivered_status' in wa.columns:
                overview['whatsapp_delivered'] = int(wa['delivered_status'].sum())
                overview['whatsapp_delivery_rate'] = round(
                    (overview['whatsapp_delivered'] / len(wa)) * 100, 2
                )
            if 'clicked_status' in wa.columns:
                overview['whatsapp_clicks'] = int(wa['clicked_status'].sum())
                overview['whatsapp_ctr'] = round(
                    (overview['whatsapp_clicks'] / len(wa)) * 100, 2
                )
        
        return jsonify(overview)
    
    except Exception as e:
        logger.error(f"Overview error: {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/metrics/campaigns", methods=["GET"])
def campaign_metrics():
    """Enhanced campaign analytics using ALL campaign data"""
    ds = get_datasets()
    if "campaigns" not in ds:
        return jsonify({"error": "campaigns dataset not loaded"}), 404
    
    c = ds["campaigns"]
    result = {
        "by_product": {},
        "by_crop": {},
        "by_campaign": {},
        "weekly_trends": [],
        "top_campaigns": [],
        "conversion_funnel": {}
    }
    
    # By product
    if "campaign_product" in c.columns and "lead_form_submission" in c.columns:
        result["by_product"] = c.groupby("campaign_product")["lead_form_submission"].sum().sort_values(ascending=False).to_dict()
    
    # By crop
    if "campaign_crop" in c.columns and "lead_form_submission" in c.columns:
        result["by_crop"] = c.groupby("campaign_crop")["lead_form_submission"].sum().sort_values(ascending=False).to_dict()
    
    # Weekly trends
    if "week_start_date" in c.columns:
        c['week_start_date'] = pd.to_datetime(c['week_start_date'], errors='coerce')
        weekly = c.groupby('week_start_date').agg({
            'social_post_impression': 'sum',
            'landing_page_visits': 'sum',
            'lead_form_submission': 'sum'
        }).reset_index()
        weekly['week_start_date'] = weekly['week_start_date'].astype(str)
        result["weekly_trends"] = weekly.to_dict(orient='records')
    
    # Top campaigns by multiple metrics
    if "campaign_id" in c.columns:
        grp = c.groupby("campaign_id").agg({
            "landing_page_visits": "sum",
            "lead_form_submission": "sum",
            "social_post_impression": "sum"
        }).reset_index()
        
        grp["conversion_pct"] = (grp["lead_form_submission"] / grp["landing_page_visits"].replace(0, np.nan) * 100).round(2)
        grp["ctr"] = (grp["landing_page_visits"] / grp["social_post_impression"].replace(0, np.nan) * 100).round(2)
        
        top = grp.nlargest(10, "lead_form_submission").fillna(0)
        result["top_campaigns"] = top.to_dict(orient="records")
    
    # Overall conversion funnel
    result["conversion_funnel"] = {
        "impressions": int(c["social_post_impression"].sum()) if "social_post_impression" in c.columns else 0,
        "visits": int(c["landing_page_visits"].sum()) if "landing_page_visits" in c.columns else 0,
        "leads": int(c["lead_form_submission"].sum()) if "lead_form_submission" in c.columns else 0
    }
    
    return jsonify(result)


@app.route("/api/metrics/growers", methods=["GET"])
def grower_metrics():
    """Comprehensive grower segmentation using ALL grower data"""
    ds = get_datasets()
    if "growers" not in ds:
        return jsonify({"error": "growers dataset not loaded"}), 404
    
    g = ds["growers"]
    state_filter = request.args.get("state")
    
    if state_filter and "state" in g.columns:
        g = g[g["state"] == state_filter]
    
    result = {
        "total": len(g),
        "state_filter": state_filter,
        "demographics": {},
        "engagement": {},
        "geographic": {},
        "technology": {}
    }
    
    # Demographics
    if "gender" in g.columns:
        result["demographics"]["gender"] = g["gender"].value_counts().to_dict()
    
    if "grower_age" in g.columns:
        bins = [0, 25, 35, 45, 55, 65, 100]
        labels = ["<25", "25-34", "35-44", "45-54", "55-64", "65+"]
        g["age_bucket"] = pd.cut(g["grower_age"], bins=bins, labels=labels, right=False)
        result["demographics"]["age_buckets"] = g["age_bucket"].value_counts().to_dict()
        result["demographics"]["avg_age"] = round(float(g["grower_age"].mean()), 1)
    
    if "grower_farm_size" in g.columns:
        bins = [0, 2, 5, 10, 25, 50, 10000]
        labels = ["<2ac", "2-5ac", "5-10ac", "10-25ac", "25-50ac", "50ac+"]
        g["farm_bucket"] = pd.cut(g["grower_farm_size"], bins=bins, labels=labels, right=False)
        result["demographics"]["farm_size_buckets"] = g["farm_bucket"].value_counts().to_dict()
        result["demographics"]["avg_farm_size"] = round(float(g["grower_farm_size"].mean()), 2)
        result["demographics"]["total_acreage"] = round(float(g["grower_farm_size"].sum()), 2)
    
    # Engagement metrics
    if "product_scan" in g.columns:
        result["engagement"]["product_scans"] = int(g["product_scan"].sum())
        result["engagement"]["scan_rate"] = round((g["product_scan"].sum() / len(g)) * 100, 2)
    
    if "offline_campaign_attended" in g.columns:
        result["engagement"]["campaign_attendees"] = int(g["offline_campaign_attended"].sum())
        result["engagement"]["attendance_rate"] = round((g["offline_campaign_attended"].sum() / len(g)) * 100, 2)
    
    if "product_name" in g.columns:
        result["engagement"]["top_products_scanned"] = g["product_name"].value_counts().head(10).to_dict()
    
    # Geographic distribution
    if "state" in g.columns:
        result["geographic"]["states"] = g["state"].value_counts().to_dict()
    
    if "district" in g.columns:
        result["geographic"]["top_districts"] = g["district"].value_counts().head(15).to_dict()
    
    if "tehsil" in g.columns:
        result["geographic"]["tehsil_count"] = len(g["tehsil"].unique())
    
    # Technology adoption
    if "language" in g.columns:
        result["technology"]["languages"] = g["language"].value_counts().to_dict()
    
    if "device_type" in g.columns:
        result["technology"]["devices"] = g["device_type"].value_counts().to_dict()
        smartphone_pct = (g["device_type"] == "smartphone").sum() / len(g) * 100
        result["technology"]["smartphone_penetration"] = round(smartphone_pct, 2)
    
    return jsonify(result)

@app.route("/api/metrics/geospatial", methods=["GET"])
def geospatial_metrics():
    """Return mock geospatial data for Indian regions to populate Map View"""
    try:
        # Generate some intelligent mock data spanning major agricultural states in India
        # Threats (red glowing spots)
        threats = [
            {"lat": 30.900965, "lng": 75.857277, "type": "pest", "label": "Yellow Rust", "intensity": 88, "state": "Punjab", "crop": "Wheat"},
            {"lat": 16.506174, "lng": 80.648015, "type": "pest", "label": "BPH Outbreak", "intensity": 95, "state": "Andhra Pradesh", "crop": "Rice"},
            {"lat": 19.751480, "lng": 75.713888, "type": "weather", "label": "Drought Stress", "intensity": 75, "state": "Maharashtra", "crop": "Cotton"},
            {"lat": 23.259933, "lng": 77.412613, "type": "disease", "label": "Late Blight", "intensity": 82, "state": "Madhya Pradesh", "crop": "Potato"},
            {"lat": 26.846708, "lng": 80.946159, "type": "pest", "label": "Fall Armyworm", "intensity": 65, "state": "Uttar Pradesh", "crop": "Corn"}
        ]
        
        # Campaigns (blue/teal pins)
        campaigns = [
            {"lat": 31.326015, "lng": 75.576180, "label": "Amistar Top Advisory", "reach": 15200, "state": "Punjab"},
            {"lat": 17.385044, "lng": 78.486671, "label": "Virtako Launch", "reach": 28400, "state": "Telangana"},
            {"lat": 22.258652, "lng": 71.192380, "label": "Kavach Promotion", "reach": 12100, "state": "Gujarat"},
            {"lat": 12.971599, "lng": 77.594566, "label": "Alika Pre-monsoon", "reach": 8500, "state": "Karnataka"}
        ]
        
        return jsonify({
            "status": "success",
            "threats": threats,
            "campaigns": campaigns
        })
    except Exception as e:
        logger.error(f"Geospatial error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/metrics/retailers", methods=["GET"])
def retailer_metrics():
    """Comprehensive retailer analytics using ALL retailer data"""
    ds = get_datasets()
    
    result = {
        "overview": {},
        "sales_performance": {},
        "inventory_health": {},
        "visit_activity": {}
    }
    
    # Overview from retailers master
    if "retailers" in ds:
        retailers = ds["retailers"]
        result["overview"]["total_retailers"] = len(retailers)
        
        if "state" in retailers.columns:
            result["overview"]["states_covered"] = len(retailers["state"].unique())
            result["overview"]["by_state"] = retailers["state"].value_counts().to_dict()
        
        if "district" in retailers.columns:
            result["overview"]["districts_covered"] = len(retailers["district"].unique())
    
    # Sales performance from POS data
    if "retailer_pos" in ds:
        pos = ds["retailer_pos"]
        
        if "sku_price" in pos.columns and "sku_qty" in pos.columns:
            pos["revenue"] = pos["sku_price"] * pos["sku_qty"]
            result["sales_performance"]["total_revenue"] = float(pos["revenue"].sum())
            result["sales_performance"]["total_units_sold"] = int(pos["sku_qty"].sum())
            result["sales_performance"]["avg_transaction_value"] = float(pos["revenue"].mean())
        
        if "retailer_id" in pos.columns:
            retailer_sales = pos.groupby("retailer_id")["revenue"].sum() if "revenue" in pos.columns else None
            if retailer_sales is not None:
                result["sales_performance"]["top_retailers"] = retailer_sales.nlargest(10).to_dict()
        
        if "sku_name" in pos.columns:
            product_sales = pos.groupby("sku_name")["sku_qty"].sum().nlargest(15)
            result["sales_performance"]["top_products"] = product_sales.to_dict()
        
        if "transaction_date" in pos.columns:
            pos["transaction_date"] = pd.to_datetime(pos["transaction_date"], errors='coerce')
            recent_sales = pos[pos["transaction_date"] >= datetime.now() - timedelta(days=30)]
            result["sales_performance"]["sales_last_30_days"] = len(recent_sales)
    
    # Inventory health
    if "retailer_inventory" in ds:
        inv = ds["retailer_inventory"]
        result["inventory_health"]["total_sku_records"] = len(inv)
        
        if "sku_qty" in inv.columns:
            out_of_stock = (inv["sku_qty"] == 0).sum()
            result["inventory_health"]["out_of_stock_instances"] = int(out_of_stock)
            result["inventory_health"]["stock_availability_rate"] = round(
                ((len(inv) - out_of_stock) / len(inv)) * 100, 2
            )
        
        if "sku_name" in inv.columns:
            result["inventory_health"]["unique_skus"] = len(inv["sku_name"].unique())
    
    # Visit activity
    if "retailer_visits" in ds:
        visits = ds["retailer_visits"]
        result["visit_activity"]["total_visits"] = len(visits)
        
        if "visit_type" in visits.columns:
            result["visit_activity"]["by_type"] = visits["visit_type"].value_counts().to_dict()
        
        if "product_recommended" in visits.columns:
            result["visit_activity"]["top_recommended_products"] = visits["product_recommended"].value_counts().head(10).to_dict()
        
        if "visit_date" in visits.columns:
            visits["visit_date"] = pd.to_datetime(visits["visit_date"], errors='coerce')
            recent_visits = visits[visits["visit_date"] >= datetime.now() - timedelta(days=30)]
            result["visit_activity"]["visits_last_30_days"] = len(recent_visits)
    
    return jsonify(result)


# ─── Urgency Detection (unchanged but using comprehensive data) ────────────────
@app.route("/api/urgency/detect", methods=["POST"])
def detect_urgency():
    """Real urgency detection from comprehensive CSV data analysis"""
    body = request.get_json(force=True)
    state = body.get("state")
    crop = body.get("crop")
    product = body.get("product")
    
    try:
        detector = get_urgency_detector()
        ds = get_datasets()
        
        result = detector.detect(
            datasets=ds,
            state_filter=state,
            crop_filter=crop,
            product_filter=product
        )
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Urgency detection error: {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/urgency/bulk", methods=["GET"])
def bulk_urgency():
    """Scan all states/crops for urgency signals using comprehensive data"""
    try:
        detector = get_urgency_detector()
        ds = get_datasets()
        result = detector.bulk_scan(ds)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Bulk urgency error: {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


# ─── Campaign Message Generation (using Qwen2.5) ──────────────────────────────
@app.route("/api/campaign/generate", methods=["POST"])
def generate_campaign():
    """
    Generate multilingual SMS + audio scripts using Qwen2.5.
    Grounded in actual grower language distribution from comprehensive CSV data.
    """
    body = request.get_json(force=True)
    campaign_type = body.get("campaign_type", "product_launch")
    product = body.get("product", "")
    crop = body.get("crop", "")
    target_state = body.get("state", "")
    languages = body.get("languages", [])
    custom_context = body.get("context", "")
    
    try:
        ds = get_datasets()
        
        # Auto-detect languages from comprehensive grower data
        if not languages and "growers" in ds:
            g = ds["growers"]
            filters = pd.Series([True] * len(g))
            if target_state and "state" in g.columns:
                filters &= g["state"] == target_state
            if crop and "grower_crop_calendar" in g.columns:
                # Parse JSON crop calendar
                filters &= g["grower_crop_calendar"].str.contains(crop, case=False, na=False)
            
            filtered = g[filters]
            if "language" in filtered.columns and len(filtered) > 0:
                lang_dist = filtered["language"].value_counts()
                languages = lang_dist.head(5).index.tolist()
        
        if not languages:
            languages = ["Hindi", "English", "Marathi"]
        
        # Get comprehensive segment statistics
        segment_stats = {}
        if "growers" in ds:
            g = ds["growers"]
            if target_state and "state" in g.columns:
                seg = g[g["state"] == target_state]
            else:
                seg = g
            
            segment_stats = {
                "total_growers": len(seg),
                "avg_farm_size": round(float(seg["grower_farm_size"].mean()), 1) if "grower_farm_size" in seg.columns else None,
                "dominant_device": seg["device_type"].mode()[0] if "device_type" in seg.columns and len(seg) > 0 and not seg["device_type"].mode().empty else "unknown",
                "smartphone_pct": round((seg["device_type"] == "smartphone").sum() / len(seg) * 100, 2) if "device_type" in seg.columns else 0,
                "avg_age": round(float(seg["grower_age"].mean()), 1) if "grower_age" in seg.columns else None,
                "language_distribution": seg["language"].value_counts().to_dict() if "language" in seg.columns else {}
            }
        
        # Use local Qwen2.5 for message generation
        messages = {}
        for lang in languages:
            prompt = f"""Generate a {campaign_type} campaign message for:
Product: {product}
Crop: {crop}
Target State: {target_state}
Language: {lang}
Additional Context: {custom_context}

Segment Statistics: {json.dumps(segment_stats)}

Generate:
1. SMS message (max 160 characters, culturally appropriate for {lang})
2. Audio script (30-45 seconds, engaging tone)

Format as JSON:
{{
    "sms": "...",
    "audio_script": "...",
    "script": "{lang} ({product})",
    "char_count": ...,
    "sms_parts": ...,
    "estimated_audio_duration_sec": ...
}}"""
            
            response = call_qwen_api(prompt, max_tokens=1000)
            
            try:
                # Try to parse JSON response
                msg_data = json.loads(response)
            except:
                # If not JSON, create structured response
                msg_data = {
                    "sms": response[:160],
                    "audio_script": response,
                    "script": f"{lang} ({product})",
                    "char_count": len(response[:160]),
                    "sms_parts": (len(response[:160]) // 160) + 1,
                    "estimated_audio_duration_sec": 30
                }
            
            messages[lang] = msg_data
        
        return jsonify({
            "campaign_type": campaign_type,
            "product": product,
            "crop": crop,
            "state": target_state,
            "languages_generated": languages,
            "segment_stats": segment_stats,
            "messages": messages
        })
    
    except Exception as e:
        logger.error(f"Campaign gen error: {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


# ─── SMS Messaging Endpoints ───────────────────────────────────────────────────
@app.route("/api/sms/send", methods=["POST"])
def send_sms():
    """
    Send a single SMS message via Twilio.
    Body: { "to_number": "+91XXXXXXXXXX", "message": "Your message here" }
    """
    body = request.get_json(force=True)
    to_number = body.get("to_number")
    message_text = body.get("message")
    
    if not to_number or not message_text:
        return jsonify({"error": "to_number and message are required"}), 400
    
    try:
        twilio = get_twilio_gateway()
        if not twilio:
            return jsonify({"error": "Twilio gateway not configured"}), 503
        
        result = twilio.send_sms(to_number=to_number, message_text=message_text)
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"SMS send error: {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


def extract_phone_numbers_from_df(df: pd.DataFrame) -> List[Dict]:
    """
    Extract phone numbers and optional metadata from a Pandas DataFrame or CSV.
    """
    import re
    phone_cols = [c for c in df.columns if any(k in c.lower() for k in ['phone', 'mobile', 'contact', 'number', 'cell', 'whatsapp', 'num'])]
    recipients = []
    seen = set()
    
    for idx, row in df.iterrows():
        raw_num = None
        if phone_cols:
            raw_num = str(row[phone_cols[0]])
        else:
            for col in df.columns:
                val = str(row[col])
                match = re.search(r'(?:\+?91[\s-]?)?([6-9]\d{9})', val)
                if match:
                    raw_num = match.group(1)
                    break
        
        if raw_num and str(raw_num).lower() not in ('nan', 'none', ''):
            digits = ''.join(c for c in str(raw_num) if c.isdigit())
            if len(digits) >= 10:
                clean_num = digits[-10:]
                if clean_num not in seen:
                    seen.add(clean_num)
                    recipients.append({
                        "farmer_id": str(row.get("grower_id") or row.get("id") or f"CUST_{idx+1:04d}"),
                        "phone_number": clean_num,
                        "name": str(row.get("name") or row.get("farmer_name") or "Valued Farmer"),
                        "language": str(row.get("language") or "Hindi"),
                        "state": str(row.get("state") or ""),
                        "crop": str(row.get("crop") or ""),
                        "device_type": str(row.get("device_type") or "smartphone")
                    })
    return recipients


@app.route("/api/sms/send-campaign", methods=["POST"])
def send_campaign_sms():
    """
    Send SMS/WhatsApp campaign to multiple recipients using messaging orchestrator.
    Body: {
        "campaign_id": "CAMP_001",
        "campaign_type": "product_launch",
        "product": "Amistar Top",
        "crop": "wheat",
        "state": "Punjab",
        "languages": ["Hindi", "English"],
        "context": "Additional context",
        "custom_numbers": ["8978518496", "9876543210"],  # optional custom target list
        "test_phone_number": "8978518496"
    }
    """
    body = request.get_json(force=True)
    campaign_id = body.get("campaign_id") or f"CAMP_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    campaign_type = body.get("campaign_type", "product_launch")
    product = body.get("product", "")
    crop = body.get("crop", "")
    target_state = body.get("state", "")
    languages = body.get("languages", [])
    custom_context = body.get("context", "")
    custom_numbers = body.get("custom_numbers") or body.get("phone_numbers") or body.get("target_numbers")
    test_phone_override = body.get("test_phone_number") or DEFAULT_TEST_PHONE
    gemini_key_req = body.get("gemini_api_key") or ""
    
    try:
        camp_gen = get_campaign_gen()
        if gemini_key_req and camp_gen:
            camp_gen.update_gemini_api_key(gemini_key_req)

        orchestrator = get_messaging_orchestrator()
        if not orchestrator:
            return jsonify({"error": "Messaging orchestrator not available"}), 503
        
        target_recipients = []
        
        # 1. Check if custom phone numbers list was provided
        if custom_numbers:
            if isinstance(custom_numbers, str):
                nums = [n.strip() for n in custom_numbers.split(",") if n.strip()]
            else:
                nums = list(custom_numbers)
            
            for idx, n in enumerate(nums):
                clean = ''.join(c for c in str(n) if c.isdigit())[-10:]
                if clean:
                    target_recipients.append({
                        "farmer_id": f"CUST_{idx+1:04d}",
                        "phone_number": clean,
                        "device_type": "smartphone",
                        "language": languages[0] if languages else "Hindi"
                    })
        
        # 2. Otherwise use dataset growers
        if not target_recipients:
            ds = get_datasets()
            if "growers" in ds:
                growers = ds["growers"]
                if target_state and "state" in growers.columns:
                    growers = growers[growers["state"] == target_state]
                if crop and "grower_crop_calendar" in growers.columns:
                    growers = growers[growers["grower_crop_calendar"].str.contains(crop, case=False, na=False)]
                
                if len(growers) == 0:
                    growers = ds["growers"].head(20)
                
                for idx_pos, (_, farmer) in enumerate(growers.iterrows()):
                    p_num = farmer.get("phone_number")
                    if not p_num or str(p_num).lower() in ("nan", "none", ""):
                        if idx_pos == 0 and test_phone_override:
                            p_num = test_phone_override
                        else:
                            p_num = f"9198765{idx_pos:05d}"
                    
                    target_recipients.append({
                        "farmer_id": str(farmer.get("grower_id", f"GRW_{idx_pos:05d}")),
                        "phone_number": p_num,
                        "device_type": str(farmer.get("device_type", "smartphone")),
                        "language": str(farmer.get("language", "Hindi"))
                    })
        
        if not target_recipients:
            target_recipients = [{
                "farmer_id": "GRW_00001",
                "phone_number": test_phone_override or "8978518496",
                "device_type": "smartphone",
                "language": "Hindi"
            }]
        
        if not languages:
            languages = ["Hindi", "English"]
        
        # Generate messages for each language using Gemini API (or templates)
        campaign_gen = get_campaign_gen()
        segment_stats = {
            "total_growers": len(target_recipients),
            "dominant_device": "smartphone"
        }
        
        messages = campaign_gen.generate_multilingual(
            campaign_type=campaign_type,
            product=product,
            crop=crop,
            state=target_state,
            languages=languages,
            context=custom_context,
            segment_stats=segment_stats
        )
        
        results = {
            "campaign_id": campaign_id,
            "total_farmers": len(target_recipients),
            "messages_generated": list(messages.keys()),
            "messages": messages,
            "product": product,
            "crop": crop,
            "state": target_state,
            "delivery_results": []
        }
        
        product_slug = product.lower().replace(' ', '_') if product else "lumina-board_care"
        image_url = f"https://lumina-board-crop-care.s3.amazonaws.com/products/{product_slug}_banner.jpg"
        
        for recipient in target_recipients:
            phone_number = recipient["phone_number"]
            farmer_lang = recipient.get("language", languages[0])
            if farmer_lang not in messages:
                farmer_lang = languages[0] if languages else "Hindi"
            
            farmer_context = {
                "farmer_id": recipient["farmer_id"],
                "phone_number": phone_number,
                "device_type": recipient.get("device_type", "smartphone"),
                "connectivity_level": "medium",
                "language": farmer_lang
            }
            
            message_content = {
                "type": campaign_type,
                "text": messages.get(farmer_lang, {}).get("sms", ""),
                "media_urls": [image_url],
                "media_url": image_url
            }
            
            if not message_content["text"]:
                continue
            
            delivery_result = orchestrator.route_campaign_message(
                farmer_context=farmer_context,
                message_content=message_content,
                campaign_id=campaign_id
            )
            
            results["delivery_results"].append({
                "farmer_id": farmer_context["farmer_id"],
                "phone_number": phone_number,
                "channel": delivery_result.get("channel", "unknown"),
                "success": delivery_result.get("success", False),
                "message_id": delivery_result.get("message_id"),
                "error": delivery_result.get("error")
            })
        
        successful = sum(1 for r in results["delivery_results"] if r["success"])
        results["summary"] = {
            "total_attempted": len(results["delivery_results"]),
            "successful": successful,
            "failed": len(results["delivery_results"]) - successful,
            "success_rate": round(successful / len(results["delivery_results"]) * 100, 2) if results["delivery_results"] else 0
        }
        
        return jsonify(results)
    
    except Exception as e:
        logger.error(f"Campaign SMS error: {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/sms/send-campaign-csv", methods=["POST"])
def send_campaign_csv():
    """
    Upload a CSV file containing phone numbers to dispatch a campaign to all CSV recipients.
    Accepts multipart form-data:
      - file: CSV file upload
      - product, crop, state, campaign_type, languages (optional form fields)
    """
    try:
        if 'file' not in request.files and 'csv' not in request.files:
            return jsonify({"error": "CSV file upload is required (field name 'file' or 'csv')"}), 400
        
        file = request.files.get('file') or request.files.get('csv')
        filename = file.filename
        
        # Read uploaded CSV
        import pandas as pd
        df = pd.read_csv(file)
        
        recipients = extract_phone_numbers_from_df(df)
        if not recipients:
            return jsonify({"error": "No valid 10-digit phone numbers found in the uploaded CSV file"}), 400
        
        # Form values
        product = request.form.get("product", "")
        crop = request.form.get("crop", "")
        state = request.form.get("state", "")
        campaign_type = request.form.get("campaign_type", "product_launch")
        languages_raw = request.form.get("languages", "")
        languages = [l.strip() for l in languages_raw.split(",") if l.strip()] if languages_raw else ["Hindi", "English"]
        campaign_id = f"CSV_CAMP_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        orchestrator = get_messaging_orchestrator()
        campaign_gen = get_campaign_gen()
        
        # Generate messages
        messages = campaign_gen.generate_multilingual(
            campaign_type=campaign_type,
            product=product,
            crop=crop,
            state=state,
            languages=languages,
            context=f"CSV Bulk Campaign uploaded from {filename} ({len(recipients)} numbers)"
        )
        
        results = {
            "campaign_id": campaign_id,
            "filename": filename,
            "phone_numbers_extracted": len(recipients),
            "messages_generated": list(messages.keys()),
            "delivery_results": []
        }
        
        product_slug = product.lower().replace(' ', '_') if product else "lumina-board_care"
        image_url = f"https://lumina-board-crop-care.s3.amazonaws.com/products/{product_slug}_banner.jpg"
        
        for r in recipients:
            p_num = r["phone_number"]
            f_lang = r.get("language")
            if f_lang not in messages:
                f_lang = languages[0]
            
            farmer_ctx = {
                "farmer_id": r["farmer_id"],
                "phone_number": p_num,
                "device_type": r.get("device_type", "smartphone"),
                "language": f_lang
            }
            msg_content = {
                "type": campaign_type,
                "text": messages.get(f_lang, {}).get("sms", ""),
                "media_urls": [image_url],
                "media_url": image_url
            }
            
            res = orchestrator.route_campaign_message(farmer_ctx, msg_content, campaign_id)
            results["delivery_results"].append({
                "farmer_id": r["farmer_id"],
                "phone_number": p_num,
                "channel": res.get("channel", "unknown"),
                "success": res.get("success", False),
                "message_id": res.get("message_id")
            })
        
        successful = sum(1 for r in results["delivery_results"] if r["success"])
        results["summary"] = {
            "filename": filename,
            "total_attempted": len(results["delivery_results"]),
            "successful": successful,
            "failed": len(results["delivery_results"]) - successful,
            "success_rate": round(successful / len(results["delivery_results"]) * 100, 2) if results["delivery_results"] else 0
        }
        
        return jsonify(results)
    
    except Exception as e:
        logger.error(f"CSV Campaign dispatch error: {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/sms/webhook", methods=["POST"])
def twilio_webhook():
    """
    Twilio webhook for delivery status callbacks.
    Configure this URL in Twilio console: https://your-domain.com/api/sms/webhook
    """
    try:
        # Twilio sends form data
        message_sid = request.form.get("MessageSid")
        message_status = request.form.get("MessageStatus")
        to_number = request.form.get("To")
        from_number = request.form.get("From")
        error_code = request.form.get("ErrorCode")
        error_message = request.form.get("ErrorMessage")
        
        logger.info(f"Twilio webhook: SID={message_sid}, Status={message_status}, To={to_number}")
        
        # Here you could update delivery logs in database
        # For now, just log and acknowledge
        
        return jsonify({"status": "received"}), 200
    
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/sms/status/<campaign_id>", methods=["GET"])
def get_sms_campaign_status(campaign_id):
    """Get delivery status for an SMS campaign"""
    try:
        orchestrator = get_messaging_orchestrator()
        if not orchestrator:
            return jsonify({"error": "Messaging orchestrator not available"}), 503
        
        status = orchestrator.get_delivery_status(campaign_id)
        return jsonify(status)
    
    except Exception as e:
        logger.error(f"Status check error: {e}")
        return jsonify({"error": str(e)}), 500


# ─── RAG Index Status ─────────────────────────────────────────────────────────
@app.route("/api/rag/status", methods=["GET"])
def rag_status():
    try:
        rag = get_rag_engine()
        return jsonify(rag.get_status())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/rag/rebuild", methods=["POST"])
def rag_rebuild():
    global _rag_engine
    _rag_engine = None
    try:
        rag = get_rag_engine()
        return jsonify({"status": "rebuilt", **rag.get_status()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─── New Automated & Free Messaging Endpoints ─────────────────────────────────

@app.route("/api/messaging/status", methods=["GET"])
def messaging_status():
    """Get status of all messaging channels and config"""
    try:
        orchestrator = get_messaging_orchestrator()
        dispatcher = get_auto_dispatcher()
        fast2sms = get_fast2sms_gateway()
        camp_gen = get_campaign_gen()
        
        status = orchestrator.check_channel_status() if orchestrator else {}
        status["default_test_phone"] = DEFAULT_TEST_PHONE
        status["auto_dispatcher_running"] = dispatcher._thread.is_alive() if (dispatcher and dispatcher._thread) else False
        status["auto_dispatch_enabled"] = dispatcher.auto_dispatch_enabled if dispatcher else False
        status["fast2sms_configured"] = bool(fast2sms and fast2sms.api_key and not fast2sms.simulate) if fast2sms else False
        status["gemini_api_configured"] = bool(camp_gen and camp_gen.gemini_api_key) if camp_gen else False
        
        return jsonify(status)
    except Exception as e:
        logger.error(f"Messaging status error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/messaging/history", methods=["GET"])
def messaging_history():
    """Get recent messaging delivery logs"""
    try:
        limit = int(request.args.get("limit", 100))
        orchestrator = get_messaging_orchestrator()
        if not orchestrator:
            return jsonify({"error": "Orchestrator unavailable"}), 503
        
        logs = orchestrator.get_all_delivery_logs(limit=limit)
        return jsonify({"count": len(logs), "logs": logs})
    except Exception as e:
        logger.error(f"Messaging history error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/messaging/config", methods=["POST"])
def messaging_config():
    """
    Update gateway configuration dynamically (e.g. Fast2SMS API key, Gemini API key, test phone number)
    Body: { "fast2sms_api_key": "...", "gemini_api_key": "...", "test_phone_number": "8978518496" }
    """
    global DEFAULT_TEST_PHONE
    try:
        body = request.get_json(force=True)
        fast2sms_key = body.get("fast2sms_api_key")
        gemini_key = body.get("gemini_api_key")
        test_phone = body.get("test_phone_number")
        
        if test_phone:
            DEFAULT_TEST_PHONE = str(test_phone).strip()
            logger.info(f"Updated default test phone number to {DEFAULT_TEST_PHONE}")
        
        fast2sms = get_fast2sms_gateway()
        if fast2sms_key is not None and fast2sms:
            fast2sms.update_api_key(fast2sms_key)
            
        camp_gen = get_campaign_gen()
        if gemini_key is not None and camp_gen:
            camp_gen.update_gemini_api_key(gemini_key)
            
        return jsonify({
            "status": "updated",
            "default_test_phone": DEFAULT_TEST_PHONE,
            "fast2sms_live_mode": bool(fast2sms and not fast2sms.simulate) if fast2sms else False,
            "gemini_api_active": bool(camp_gen and camp_gen.gemini_api_key) if camp_gen else False
        })
    except Exception as e:
        logger.error(f"Config update error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/messaging/auto-dispatch", methods=["POST"])
def trigger_auto_dispatch():
    """
    Trigger immediate background/manual bio-urgency scanning and customer message auto-dispatch
    Body: { "min_urgency_score": 65.0, "target_test_number": "8978518496" }
    """
    try:
        body = request.get_json(force=True) or {}
        min_score = float(body.get("min_urgency_score", 65.0))
        target_number = body.get("target_test_number") or DEFAULT_TEST_PHONE
        
        dispatcher = get_auto_dispatcher()
        if not dispatcher:
            return jsonify({"error": "Auto dispatcher unavailable"}), 503
        
        res = dispatcher.run_manual_auto_dispatch(min_urgency_score=min_score, target_test_number=target_number)
        return jsonify(res)
    except Exception as e:
        logger.error(f"Auto dispatch error: {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/sms/send-free", methods=["POST"])
def send_free_sms():
    """
    Send a direct free SMS message via Fast2SMS / ADB / Simulation
    Body: { "to_number": "8978518496", "message": "Text..." }
    """
    try:
        body = request.get_json(force=True)
        to_number = body.get("to_number") or DEFAULT_TEST_PHONE
        message_text = body.get("message")
        
        if not message_text:
            return jsonify({"error": "message is required"}), 400
        
        orchestrator = get_messaging_orchestrator()
        if not orchestrator:
            return jsonify({"error": "Orchestrator unavailable"}), 503
        
        res = orchestrator.send_sms(to_number=to_number, message_text=message_text, campaign_id="DIRECT_FREE_SMS")
        return jsonify(res)
    except Exception as e:
        logger.error(f"Send free SMS error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/whatsapp/send-free", methods=["POST"])
def send_free_whatsapp():
    """
    Send a WhatsApp message with optional image URL
    Body: { "to_number": "8978518496", "message": "Text...", "image_url": "https://..." }
    """
    try:
        body = request.get_json(force=True)
        to_number = body.get("to_number") or DEFAULT_TEST_PHONE
        message_text = body.get("message")
        image_url = body.get("image_url")
        
        if not message_text:
            return jsonify({"error": "message is required"}), 400
        
        orchestrator = get_messaging_orchestrator()
        if not orchestrator:
            return jsonify({"error": "Orchestrator unavailable"}), 503
        
        res = orchestrator.send_whatsapp(
            to_number=to_number,
            message_text=message_text,
            campaign_id="DIRECT_FREE_WA",
            media_url=image_url
        )
        # Also provide direct wa.me link for browser 1-click send
        clean_num = ''.join(c for c in str(to_number) if c.isdigit())
        if len(clean_num) == 10:
            clean_num = '91' + clean_num
        import urllib.parse
        encoded_msg = urllib.parse.quote(message_text + (f"\n\n📷 Visual Banner: {image_url}" if image_url else ""))
        res["wa_me_link"] = f"https://wa.me/{clean_num}?text={encoded_msg}"
        
        return jsonify(res)
    except Exception as e:
        logger.error(f"Send free WhatsApp error: {e}")
        return jsonify({"error": str(e)}), 500


# ─── Vernacular Audio Synthesis Endpoint (gTTS + Robust Fallbacks) ─────────────
LANG_TO_GTTS = {
    "hindi": "hi",
    "telugu": "te",
    "marathi": "mr",
    "punjabi": "pa",
    "tamil": "ta",
    "kannada": "kn",
    "bengali": "bn",
    "gujarati": "gu",
    "odia": "hi",  # Fallback to Hindi if Odia requested in gTTS
    "malayalam": "ml",
    "english": "en",
    "hi": "hi", "te": "te", "mr": "mr", "pa": "pa", "ta": "ta",
    "kn": "kn", "bn": "bn", "gu": "gu", "or": "hi", "ml": "ml", "en": "en"
}


def generate_audio_stream_data(text: str, lang_name: str):
    """
    Generate audio stream (MP3/WAV) using gTTS with robust multi-tiered fallback:
    1. Primary gTTS language mapping
    2. Fallback gTTS Hindi ('hi')
    3. Fallback gTTS English ('en')
    4. In-memory synthetic WAV audio generator (guarantees audio stream never fails/returns 500 error)
    """
    import io
    import re
    clean_text = re.sub(r'\[.*?\]', '', text or '').strip()
    if not clean_text or len(clean_text) < 2:
        clean_text = f"Lumina Board Vernacular Voice Advisory in {lang_name or 'Hindi'}"

    lang_key = (lang_name or "Hindi").lower().strip()
    lang_code = LANG_TO_GTTS.get(lang_key, "hi")

    # Tier 1: Primary gTTS
    try:
        from gtts import gTTS
        tts = gTTS(text=clean_text, lang=lang_code, slow=False)
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        return mp3_fp, "audio/mpeg"
    except Exception as e1:
        logger.warning(f"gTTS primary lang ({lang_code}) error: {e1}")

    # Tier 2: Fallback gTTS Hindi
    if lang_code != "hi":
        try:
            from gtts import gTTS
            tts = gTTS(text=clean_text, lang="hi", slow=False)
            mp3_fp = io.BytesIO()
            tts.write_to_fp(mp3_fp)
            mp3_fp.seek(0)
            return mp3_fp, "audio/mpeg"
        except Exception as e2:
            logger.warning(f"gTTS Hindi fallback error: {e2}")

    # Tier 3: Fallback gTTS English
    if lang_code != "en":
        try:
            from gtts import gTTS
            tts = gTTS(text=clean_text, lang="en", slow=False)
            mp3_fp = io.BytesIO()
            tts.write_to_fp(mp3_fp)
            mp3_fp.seek(0)
            return mp3_fp, "audio/mpeg"
        except Exception as e3:
            logger.warning(f"gTTS English fallback error: {e3}")

    # Tier 4: Pure Python synthetic WAV tone fallback (offline/network failure)
    import math
    import struct
    import wave
    wav_io = io.BytesIO()
    with wave.open(wav_io, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(22050)
        num_samples = int(2.5 * 22050)
        for i in range(num_samples):
            t = i / 22050
            decay = math.exp(-t * 1.2)
            sample = int(16000 * decay * (math.sin(2 * math.pi * 440 * t) + 0.5 * math.sin(2 * math.pi * 660 * t)))
            sample = max(-32768, min(32767, sample))
            wav_file.writeframes(struct.pack('<h', sample))
    wav_io.seek(0)
    return wav_io, "audio/wav"


@app.route("/api/audio/stream", methods=["GET", "POST"])
def stream_audio():
    """
    Stream vernacular speech MP3 for direct inline HTML5 <audio> tag playback
    Query params or JSON: text, language
    """
    try:
        if request.method == "POST":
            body = request.get_json(force=True, silent=True) or {}
            text = body.get("text", "")
            lang_name = (body.get("language") or "Hindi").lower().strip()
        else:
            text = request.args.get("text", "")
            lang_name = (request.args.get("language") or "Hindi").lower().strip()

        audio_fp, mimetype = generate_audio_stream_data(text, lang_name)
        ext = "wav" if mimetype == "audio/wav" else "mp3"
        return send_file(
            audio_fp,
            mimetype=mimetype,
            as_attachment=False,
            download_name=f"stream.{ext}"
        )
    except Exception as e:
        logger.error(f"Audio stream unexpected error: {e}\n{traceback.format_exc()}")
        audio_fp, mimetype = generate_audio_stream_data("Lumina Board Advisory", "Hindi")
        ext = "wav" if mimetype == "audio/wav" else "mp3"
        return send_file(audio_fp, mimetype=mimetype, as_attachment=False, download_name=f"stream.{ext}")


@app.route("/api/audio/synthesize", methods=["POST"])
def synthesize_audio():
    """
    Synthesize vernacular speech MP3 from IVR text using gTTS
    Body: { "text": "...", "language": "Hindi" }
    """
    try:
        import time
        body = request.get_json(force=True, silent=True) or {}
        text = body.get("text", "")
        lang_name = (body.get("language") or "Hindi").lower().strip()

        audio_fp, mimetype = generate_audio_stream_data(text, lang_name)
        ext = "wav" if mimetype == "audio/wav" else "mp3"

        return send_file(
            audio_fp,
            mimetype=mimetype,
            as_attachment=True,
            download_name=f"lumina_advisory_{lang_name}_{int(time.time())}.{ext}"
        )
    except Exception as e:
        logger.error(f"Audio synthesis unexpected error: {e}\n{traceback.format_exc()}")
        audio_fp, mimetype = generate_audio_stream_data("Lumina Board Advisory", "Hindi")
        return send_file(audio_fp, mimetype=mimetype, as_attachment=True, download_name="lumina_advisory.wav")


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    debug = os.environ.get("DEBUG", "true").lower() == "true"
    logger.info(f"Starting Lumina Board Enhanced API on port {port}")
    logger.info(f"Qwen2.5 API: {QWEN_API_URL}")
    logger.info(f"Model: {QWEN_MODEL}")
    app.run(host="0.0.0.0", port=port, debug=debug)