"""
WhatsApp Messaging Channel
Supports three modes:
  1. ADB (Android Debug Bridge) - Control physical Android phone
  2. WhatsApp Business Cloud API - Meta's official API
  3. Simulation mode - Testing without credentials
"""
import subprocess
import time
import json
import uuid
import logging
import re
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger("lumina.messaging.whatsapp")


class WhatsAppChannel:
    """
    WhatsApp messaging channel with multiple backends.
    
    Modes:
      - adb: Control WhatsApp via Android Debug Bridge (physical phone)
      - business_api: WhatsApp Business Cloud API (Meta)
      - simulate: Mock mode for testing without credentials
    """

    def __init__(
        self,
        mode: str = "simulate",
        device_id: str = None,
        access_token: str = None,
        phone_number_id: str = None,
        business_account_id: str = None,
        webhook_verify_token: str = None
    ):
        """
        Initialize WhatsApp channel

        Args:
            mode: 'adb', 'business_api', or 'simulate'
            device_id: ADB device ID (for adb mode)
            access_token: WhatsApp Business API access token
            phone_number_id: WhatsApp Business API phone number ID
            business_account_id: WABA ID
            webhook_verify_token: Webhook verification token
        """
        self.mode = mode
        self.device_id = device_id
        self.access_token = access_token
        self.phone_number_id = phone_number_id
        self.business_account_id = business_account_id
        self.webhook_verify_token = webhook_verify_token

        self.sent_history = []
        self.simulation_log = []
        self.message_queue = []

        # ADB connection state
        self._adb_connected = False
        if mode == "adb":
            self._adb_connected = self._check_adb_connection()

        # Business API base URL
        self._api_base = "https://graph.facebook.com/v18.0"

        logger.info(f"WhatsApp Channel initialized in {mode.upper()} mode")

    # ─── Public API ─────────────────────────────────────────────────────────────

    def send_message(
        self,
        to_number: str,
        message_text: str,
        media_url: str = None,
        media_type: str = None,
        template_name: str = None,
        template_params: Dict = None,
        retry_attempts: int = 3
    ) -> Dict:
        """
        Send WhatsApp message via configured mode

        Args:
            to_number: Recipient phone number (with country code)
            message_text: Message content
            media_url: URL of media to attach (image, video, document, audio)
            media_type: 'image', 'video', 'document', 'audio'
            template_name: Pre-approved template name (for business_api)
            template_params: Template parameter values
            retry_attempts: Number of retries on failure

        Returns:
            Dict with success status and message details
        """
        clean_number = self._clean_phone_number(to_number)

        if self.mode == "simulate":
            return self._simulate_send(clean_number, message_text, media_url, media_type)

        if self.mode == "adb":
            return self._send_via_adb(clean_number, message_text, media_url, retry_attempts)

        if self.mode == "business_api":
            if template_name:
                return self._send_template_message(clean_number, template_name, template_params)
            return self._send_via_api(clean_number, message_text, media_url, media_type)

        return self._error_response(clean_number, f"Unknown mode: {self.mode}")

    def send_bulk(
        self,
        numbers: List[str],
        message_text: str,
        media_url: str = None,
        media_type: str = None
    ) -> Dict:
        """Send WhatsApp message to multiple recipients"""
        results = []
        for number in numbers:
            result = self.send_message(number, message_text, media_url, media_type)
            results.append(result)

        successful = sum(1 for r in results if r.get('success', False))

        return {
            'success': successful > 0,
            'total': len(numbers),
            'successful': successful,
            'failed': len(numbers) - successful,
            'results': results,
            'provider': 'whatsapp',
            'mode': self.mode
        }

    def send_template_message(
        self,
        to_number: str,
        template_name: str,
        parameters: Dict = None,
        language: str = "en"
    ) -> Dict:
        """
        Send a pre-approved WhatsApp template message
        Required for business_api mode (Meta policy)
        """
        if self.mode == "simulate":
            return self._simulate_send(to_number, f"[TEMPLATE: {template_name}] {json.dumps(parameters)}")

        if self.mode != "business_api":
            return self._error_response(to_number, "Template messages require business_api mode")

        return self._send_template_message(to_number, template_name, parameters, language)

    def send_media(
        self,
        to_number: str,
        media_url: str,
        media_type: str,
        caption: str = None
    ) -> Dict:
        """Send media message (image, video, document, audio)"""
        return self.send_message(
            to_number=to_number,
            message_text=caption or "",
            media_url=media_url,
            media_type=media_type
        )

    def get_sent_history(self, phone_number: str = None) -> List[Dict]:
        """Get message history"""
        if phone_number:
            return [m for m in self.sent_history if m.get('phone_number') == phone_number]
        return self.sent_history

    def get_simulation_log(self) -> List[Dict]:
        """Get simulation log entries"""
        return self.simulation_log

    def check_connection(self) -> Dict:
        """Check if the channel is connected and operational"""
        if self.mode == "simulate":
            return {'connected': True, 'mode': 'simulation', 'note': 'Simulation mode - always connected'}

        if self.mode == "adb":
            return {
                'connected': self._adb_connected,
                'mode': 'adb',
                'device_id': self.device_id or 'auto'
            }

        if self.mode == "business_api":
            # Verify token by making a test call
            try:
                import requests
                resp = requests.get(
                    f"{self._api_base}/{self.phone_number_id}",
                    headers={"Authorization": f"Bearer {self.access_token}"},
                    timeout=10
                )
                return {
                    'connected': resp.status_code == 200,
                    'mode': 'business_api',
                    'phone_number_id': self.phone_number_id,
                    'status_code': resp.status_code
                }
            except Exception as e:
                return {'connected': False, 'mode': 'business_api', 'error': str(e)}

        return {'connected': False, 'mode': self.mode, 'error': 'Unknown mode'}

    # ─── ADB Implementation ─────────────────────────────────────────────────────

    def _check_adb_connection(self) -> bool:
        """Verify ADB connection to Android device"""
        try:
            cmd = ['adb', 'devices']
            if self.device_id:
                cmd = ['adb', '-s', self.device_id, 'devices']

            output = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            connected = 'device' in output.stdout and 'unauthorized' not in output.stdout

            if connected:
                logger.info("ADB device connected")
            else:
                logger.warning("No ADB device found. Devices:\n%s", output.stdout)

            return connected
        except FileNotFoundError:
            logger.error("ADB not found. Install Android Debug Bridge.")
            return False
        except subprocess.TimeoutExpired:
            logger.error("ADB check timed out")
            return False
        except Exception as e:
            logger.error(f"ADB check error: {e}")
            return False

    def _send_via_adb(
        self,
        phone_number: str,
        message_text: str,
        media_path: str = None,
        retry_attempts: int = 3
    ) -> Dict:
        """Send WhatsApp message via ADB (Android phone)"""
        if not self._adb_connected:
            return self._error_response(phone_number, 'ADB not connected. Run: adb devices')

        try:
            # Open WhatsApp chat
            self._adb_open_chat(phone_number)
            time.sleep(2)

            # Attach media if provided
            if media_path:
                self._adb_attach_media(media_path)
                time.sleep(1)

            # Type and send message
            self._adb_type_text(message_text)
            time.sleep(0.5)
            self._adb_tap_send()

            # Log success
            record = {
                'phone_number': phone_number,
                'message': message_text,
                'media': media_path,
                'timestamp': datetime.now().isoformat(),
                'status': 'sent',
                'mode': 'adb'
            }
            self.sent_history.append(record)

            return {
                'success': True,
                'phone_number': phone_number,
                'message_id': self._generate_message_id(),
                'timestamp': datetime.now().isoformat(),
                'provider': 'whatsapp',
                'mode': 'adb'
            }

        except subprocess.CalledProcessError as e:
            logger.error(f"ADB command failed: {e}")
            if retry_attempts > 0:
                time.sleep(2)
                return self._send_via_adb(phone_number, message_text, media_path, retry_attempts - 1)
            return self._error_response(phone_number, f"ADB error: {e}")
        except Exception as e:
            logger.error(f"ADB send error: {e}")
            if retry_attempts > 0:
                time.sleep(2)
                return self._send_via_adb(phone_number, message_text, media_path, retry_attempts - 1)
            return self._error_response(phone_number, str(e))

    def _adb_open_chat(self, phone_number: str):
        """Open WhatsApp chat with specific contact via intent"""
        device_prefix = ['adb']
        if self.device_id:
            device_prefix = ['adb', '-s', self.device_id]

        # Use WhatsApp URI scheme to open chat
        cmd = device_prefix + [
            'shell', 'am', 'start',
            '-a', 'android.intent.action.VIEW',
            '-d', f'https://wa.me/{phone_number}',
            'com.whatsapp/.ui.LauncherActivity'
        ]
        subprocess.run(cmd, check=True, timeout=10)

    def _adb_type_text(self, text: str):
        """Type text in WhatsApp chat via ADB"""
        device_prefix = ['adb']
        if self.device_id:
            device_prefix = ['adb', '-s', self.device_id]

        # Escape special characters for ADB input
        escaped_text = text.replace('"', '\\"').replace("'", "\\'")
        escaped_text = escaped_text.replace('(', '\\(').replace(')', '\\)')
        escaped_text = escaped_text.replace('&', '\\&').replace('|', '\\|')
        escaped_text = escaped_text.replace(';', '\\;').replace('<', '\\<').replace('>', '\\>')
        escaped_text = escaped_text.replace('`', '\\`').replace('$', '\\$')

        # For long messages, use multiple input commands
        max_chunk = 100
        for i in range(0, len(escaped_text), max_chunk):
            chunk = escaped_text[i:i + max_chunk]
            cmd = device_prefix + ['shell', 'input', 'text', chunk]
            subprocess.run(cmd, check=True, timeout=10)
            if i + max_chunk < len(escaped_text):
                time.sleep(0.1)

    def _adb_attach_media(self, media_path: str):
        """Attach media file in WhatsApp"""
        device_prefix = ['adb']
        if self.device_id:
            device_prefix = ['adb', '-s', self.device_id]

        # Push file to device if local
        if not media_path.startswith('/sdcard/'):
            push_cmd = device_prefix + ['push', media_path, '/sdcard/Download/']
            subprocess.run(push_cmd, check=True, timeout=30)
            media_path = f'/sdcard/Download/{media_path.split("/")[-1]}'

        # Tap attachment button (coordinates may vary by device)
        # Common positions: 1000, 1900 for many devices
        attach_cmd = device_prefix + ['shell', 'input', 'tap', '1000', '1900']
        subprocess.run(attach_cmd, check=True, timeout=5)
        time.sleep(1)

        # Tap gallery/file option
        gallery_cmd = device_prefix + ['shell', 'input', 'tap', '200', '1500']
        subprocess.run(gallery_cmd, check=True, timeout=5)
        time.sleep(2)

    def _adb_tap_send(self):
        """Tap the send button in WhatsApp"""
        device_prefix = ['adb']
        if self.device_id:
            device_prefix = ['adb', '-s', self.device_id]

        # Try to detect send button position
        # Common positions: 1000, 2000 (bottom right)
        # Alternative: Use KEYCODE_ENTER
        try:
            # First try using key event (works on most devices)
            cmd = device_prefix + ['shell', 'input', 'keyevent', 'KEYCODE_ENTER']
            subprocess.run(cmd, check=True, timeout=5)
        except:
            # Fallback: tap coordinates
            cmd = device_prefix + ['shell', 'input', 'tap', '1000', '2000']
            subprocess.run(cmd, check=True, timeout=5)

    # ─── WhatsApp Business Cloud API Implementation ─────────────────────────────

    def _send_via_api(
        self,
        to_number: str,
        message_text: str,
        media_url: str = None,
        media_type: str = None
    ) -> Dict:
        """Send message via WhatsApp Business Cloud API"""
        if not self.access_token or not self.phone_number_id:
            return self._error_response(to_number, "Business API not configured")

        try:
            import requests

            if media_url and media_type:
                # Send media message
                payload = {
                    "messaging_product": "whatsapp",
                    "recipient_type": "individual",
                    "to": to_number,
                    "type": media_type,
                    media_type: {
                        "link": media_url,
                        "caption": message_text or ""
                    }
                }
            else:
                # Send text message
                payload = {
                    "messaging_product": "whatsapp",
                    "recipient_type": "individual",
                    "to": to_number,
                    "type": "text",
                    "text": {
                        "preview_url": True,
                        "body": message_text
                    }
                }

            response = requests.post(
                f"{self._api_base}/{self.phone_number_id}/messages",
                headers={
                    "Authorization": f"Bearer {self.access_token}",
                    "Content-Type": "application/json"
                },
                json=payload,
                timeout=30
            )

            result = response.json()

            if response.status_code == 200:
                message_id = result.get("messages", [{}])[0].get("id", self._generate_message_id())
                record = {
                    'phone_number': to_number,
                    'message': message_text,
                    'message_id': message_id,
                    'status': 'sent',
                    'timestamp': datetime.now().isoformat(),
                    'mode': 'business_api'
                }
                self.sent_history.append(record)

                return {
                    'success': True,
                    'message_id': message_id,
                    'status': 'sent',
                    'timestamp': datetime.now().isoformat(),
                    'provider': 'whatsapp',
                    'mode': 'business_api'
                }
            else:
                error = result.get("error", {}).get("message", "Unknown API error")
                return self._error_response(to_number, error)

        except ImportError:
            return self._error_response(to_number, "requests library required for Business API")
        except Exception as e:
            return self._error_response(to_number, str(e))

    def _send_template_message(
        self,
        to_number: str,
        template_name: str,
        parameters: Dict = None,
        language: str = "en"
    ) -> Dict:
        """Send pre-approved template via WhatsApp Business API"""
        if not self.access_token or not self.phone_number_id:
            return self._error_response(to_number, "Business API not configured")

        try:
            import requests

            # Build template components
            components = []
            if parameters:
                body_params = []
                for key, value in parameters.items():
                    body_params.append({
                        "type": "text",
                        "text": str(value)
                    })
                if body_params:
                    components.append({
                        "type": "body",
                        "parameters": body_params
                    })

            payload = {
                "messaging_product": "whatsapp",
                "recipient_type": "individual",
                "to": to_number,
                "type": "template",
                "template": {
                    "name": template_name,
                    "language": {
                        "code": language
                    }
                }
            }

            if components:
                payload["template"]["components"] = components

            response = requests.post(
                f"{self._api_base}/{self.phone_number_id}/messages",
                headers={
                    "Authorization": f"Bearer {self.access_token}",
                    "Content-Type": "application/json"
                },
                json=payload,
                timeout=30
            )

            result = response.json()

            if response.status_code == 200:
                message_id = result.get("messages", [{}])[0].get("id", self._generate_message_id())
                return {
                    'success': True,
                    'message_id': message_id,
                    'status': 'sent',
                    'template': template_name,
                    'timestamp': datetime.now().isoformat(),
                    'provider': 'whatsapp',
                    'mode': 'business_api'
                }
            else:
                error = result.get("error", {}).get("message", "Unknown API error")
                return self._error_response(to_number, error)

        except Exception as e:
            return self._error_response(to_number, str(e))

    # ─── Simulation ─────────────────────────────────────────────────────────────

    def _simulate_send(
        self,
        phone_number: str,
        message_text: str,
        media_url: str = None,
        media_type: str = None
    ) -> Dict:
        """Simulate sending WhatsApp message"""
        time.sleep(0.3)  # Simulate network delay
        message_id = f"WA_SIM_{uuid.uuid4().hex[:12].upper()}"

        record = {
            'phone_number': phone_number,
            'message': message_text,
            'media': media_url,
            'media_type': media_type,
            'message_id': message_id,
            'status': 'sent',
            'timestamp': datetime.now().isoformat(),
            'mode': 'simulation'
        }
        self.sent_history.append(record)
        self.simulation_log.append({
            **record,
            'event': 'whatsapp_sent_simulated'
        })

        logger.info(f"[SIMULATION] WhatsApp sent to {phone_number}: {message_text[:50]}...")

        return {
            'success': True,
            'message_id': message_id,
            'phone_number': phone_number,
            'status': 'sent',
            'timestamp': datetime.now().isoformat(),
            'provider': 'whatsapp',
            'mode': 'simulation',
            'note': 'This is a simulated WhatsApp message. Configure credentials for real sending.'
        }

    # ─── Helpers ────────────────────────────────────────────────────────────────

    def _clean_phone_number(self, number: str) -> str:
        """Clean phone number to E.164 format"""
        clean = number.replace("+", "").replace(" ", "").replace("-", "").replace("(", "").replace(")", "")
        # Ensure it starts with +
        return clean

    def _generate_message_id(self) -> str:
        """Generate unique message ID"""
        return f"msg_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"

    def _error_response(self, number: str, error: str) -> Dict:
        """Build error response"""
        return {
            'success': False,
            'error': error,
            'phone_number': number,
            'provider': 'whatsapp',
            'mode': self.mode
        }


# Alias for backward compatibility
ADBWhatsAppController = WhatsAppChannel


# Example usage
if __name__ == "__main__":
    # Test simulation mode
    wa = WhatsAppChannel(mode="simulate")
    result = wa.send_message("919876543210", "नमस्ते किसान भाई! Lumina Board का नया उत्पाद अब उपलब्ध है।")
    print(json.dumps(result, indent=2))

    # Test bulk
    bulk = wa.send_bulk(["919876543210", "919876543211"], "Bulk WhatsApp test")
    print(json.dumps(bulk, indent=2))

    # Check connection
    status = wa.check_connection()
    print(json.dumps(status, indent=2))