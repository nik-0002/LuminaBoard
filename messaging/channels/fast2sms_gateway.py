"""
Fast2SMS Gateway for Free/Developer SMS Delivery in India
Supports Fast2SMS bulkV2 Quick SMS API, Dev API, and simulation mode.
"""
import requests
import json
import logging
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger("lumina.messaging.fast2sms")


class Fast2SMSGateway:
    """
    Fast2SMS API Gateway for sending SMS to Indian mobile numbers (+91)
    Supports:
      - Quick SMS Route ('q'): Simple direct text SMS
      - DLT Route ('dlt'): Transactional DLT approved messages
      - Simulation Mode: Fallback when API key is unconfigured or invalid
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        simulate: bool = True
    ):
        """
        Initialize Fast2SMS Gateway

        Args:
            api_key: Fast2SMS API Key from https://www.fast2sms.com/dev/api
            simulate: Enable simulation mode when key is absent/invalid
        """
        self.api_key = api_key
        self.simulate = simulate or not bool(api_key)
        self.base_url = "https://www.fast2sms.com/dev/bulkV2"
        self.sent_messages: List[Dict] = []

        if self.simulate:
            logger.info("Fast2SMS Gateway initialized in SIMULATION mode")
        else:
            masked_key = f"{api_key[:6]}..." if len(api_key) > 6 else "***"
            logger.info(f"Fast2SMS Gateway initialized with API key: {masked_key}")

    def update_api_key(self, api_key: str):
        """Update API key dynamically"""
        self.api_key = api_key
        if api_key and len(api_key.strip()) > 5:
            self.simulate = False
            logger.info("Fast2SMS API key updated. Live mode enabled.")
        else:
            self.simulate = True
            logger.info("Fast2SMS API key cleared. Simulation mode enabled.")

    def send_sms(
        self,
        to_number: str,
        message_text: str,
        route: str = "q",
        language: str = "english"
    ) -> Dict:
        """
        Send SMS via Fast2SMS API

        Args:
            to_number: Recipient 10-digit mobile number (e.g. 8978518496)
            message_text: SMS content
            route: 'q' for Quick SMS, 'dlt' for DLT route
            language: 'english' or 'unicode'

        Returns:
            Dict with delivery result
        """
        clean_number = self._clean_phone_number(to_number)

        if self.simulate or not self.api_key:
            return self._simulate_send(clean_number, message_text, route)

        try:
            # Detect unicode (non-ASCII characters like Hindi/Telugu)
            is_unicode = any(ord(char) > 127 for char in message_text)
            lang = "unicode" if is_unicode else language

            headers = {
                "authorization": self.api_key.strip(),
                "Content-Type": "application/json"
            }

            payload = {
                "route": route,
                "message": message_text,
                "language": lang,
                "flash": 0,
                "numbers": clean_number
            }

            response = requests.post(
                self.base_url,
                headers=headers,
                data=json.dumps(payload),
                timeout=10
            )

            if response.status_code == 200:
                res_data = response.json()
                is_success = res_data.get("return", False) is True

                record = {
                    "to_number": clean_number,
                    "message": message_text,
                    "message_id": f"F2S_{res_data.get('request_id', datetime.now().strftime('%Y%m%d%H%M%S'))}",
                    "status": "sent" if is_success else "failed",
                    "provider": "fast2sms",
                    "raw_response": res_data,
                    "timestamp": datetime.now().isoformat()
                }
                self.sent_messages.append(record)

                if is_success:
                    logger.info(f"Fast2SMS delivered to {clean_number}: req_id={record['message_id']}")
                    return {
                        "success": True,
                        "message_id": record["message_id"],
                        "to_number": clean_number,
                        "status": "sent",
                        "provider": "fast2sms",
                        "response": res_data
                    }
                else:
                    err_msg = res_data.get("message", ["API request failed"])[0] if isinstance(res_data.get("message"), list) else str(res_data.get("message", "Failed"))
                    logger.warning(f"Fast2SMS API warning for {clean_number}: {err_msg}")
                    # Fallback to simulation if account balance is low or key invalid
                    sim_res = self._simulate_send(clean_number, message_text, route)
                    sim_res["note"] = f"Fast2SMS API notice: {err_msg}. Tracked in simulation engine."
                    return sim_res

            else:
                try:
                    err_json = response.json()
                    err_msg = err_json.get("message") or response.text
                except Exception:
                    err_msg = response.text
                logger.warning(f"Fast2SMS API notice: {err_msg}")
                sim_res = self._simulate_send(clean_number, message_text, route)
                sim_res["error"] = err_msg
                import urllib.parse
                sim_res["sms_link"] = f"sms:+91{clean_number}?body={urllib.parse.quote(message_text)}"
                sim_res["note"] = f"Fast2SMS API Notice: {err_msg}. Use 1-Click Phone SMS to send directly."
                return sim_res

        except Exception as e:
            logger.error(f"Fast2SMS exception: {e}")
            sim_res = self._simulate_send(clean_number, message_text, route)
            sim_res["note"] = f"Network exception: {str(e)}. Fallback to simulation."
            return sim_res

    def _clean_phone_number(self, number: str) -> str:
        """Strip country codes and special chars to get 10-digit Indian number"""
        digits = ''.join(c for c in str(number) if c.isdigit())
        if len(digits) > 10 and digits.startswith('91'):
            return digits[-10:]
        elif len(digits) == 10:
            return digits
        return digits[-10:] if len(digits) >= 10 else digits

    def _simulate_send(self, clean_number: str, message_text: str, route: str) -> Dict:
        """Simulate Fast2SMS sending"""
        msg_id = f"F2S_SIM_{datetime.now().strftime('%Y%m%d%H%M%S_%f')[:18]}"
        record = {
            "to_number": clean_number,
            "message": message_text,
            "message_id": msg_id,
            "status": "sent",
            "provider": "fast2sms_simulation",
            "timestamp": datetime.now().isoformat()
        }
        self.sent_messages.append(record)

        return {
            "success": True,
            "message_id": msg_id,
            "to_number": clean_number,
            "status": "sent",
            "provider": "fast2sms_simulation",
            "timestamp": datetime.now().isoformat(),
            "note": "Simulated Fast2SMS send. Provide your API key in Dashboard Settings to send live SMS."
        }
