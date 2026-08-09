"""
MSG91 SMS Gateway for Indian SMS delivery
Supports both Flow API (v5) and Simple SMS API (v4)
Includes simulation mode for testing without credentials
"""
import requests
import json
import time
import uuid
import logging
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger("lumina.messaging.msg91")


class MSG91Gateway:
    """
    MSG91 API gateway for SMS messaging in India
    Supports:
      - Flow API (v5): Pre-approved templates via flow_id
      - Simple SMS API (v4): Direct message sending (route 4 = transactional, route 1 = promotional)
      - Simulation mode: Works without real credentials for testing
    """

    def __init__(
        self,
        auth_key: str = None,
        sender_id: str = "SYNGTA",
        template_id: str = None,
        use_simple_api: bool = True,
        simulate: bool = True
    ):
        """
        Initialize MSG91 gateway

        Args:
            auth_key: Your MSG91 Auth Key from dashboard (None = simulation mode)
            sender_id: 6-character sender ID (approved in MSG91 panel)
            template_id: Pre-approved template ID for transactional SMS (Flow API)
            use_simple_api: True = Simple SMS API (v4), False = Flow API (v5)
            simulate: Enable simulation mode when no credentials available
        """
        self.auth_key = auth_key
        self.sender_id = sender_id
        self.template_id = template_id
        self.use_simple_api = use_simple_api
        self.simulate = simulate or not auth_key
        self.base_url_v4 = "https://api.msg91.com/api/v4"
        self.base_url_v5 = "https://api.msg91.com/api/v5"
        self.sent_messages = []
        self.simulation_log = []

        if self.simulate:
            logger.info("MSG91 Gateway initialized in SIMULATION mode")
        else:
            logger.info(f"MSG91 Gateway initialized with auth_key={auth_key[:8]}...")

    def send_sms(
        self,
        to_number: str,
        message_text: str,
        route: str = "4",
        template_id: str = None,
        country_code: str = "91"
    ) -> Dict:
        """
        Send SMS via MSG91 API

        Args:
            to_number: Mobile number (with or without country code)
            message_text: SMS content
            route: "4" for transactional, "1" for promotional
            template_id: Template ID for Flow API (v5)
            country_code: Country code (default 91 for India)

        Returns:
            Dict with success status and message details
        """
        try:
            # Clean phone number
            clean_number = self._clean_phone_number(to_number, country_code)

            if self.simulate:
                return self._simulate_send(clean_number, message_text, route)

            if self.use_simple_api:
                return self._send_simple_sms(clean_number, message_text, route)
            else:
                return self._send_flow_sms(clean_number, message_text, route, template_id)

        except requests.exceptions.Timeout:
            return self._error_response(to_number, 'Request timeout')
        except requests.exceptions.RequestException as e:
            return self._error_response(to_number, f'Network error: {str(e)}')
        except Exception as e:
            logger.error(f"MSG91 send_sms error: {e}")
            return self._error_response(to_number, str(e))

    def _send_simple_sms(self, clean_number: str, message_text: str, route: str) -> Dict:
        """
        Send SMS via MSG91 Simple SMS API (v4)
        No template required - sends message text directly
        """
        payload = {
            "sender": self.sender_id,
            "mobiles": clean_number,
            "route": route,
            "sms": message_text,
            "unicode": 1,  # Support Unicode/Indic scripts
            "DLT_TE_ID": self.template_id or ""  # Optional DLT template ID
        }

        headers = {
            "authkey": self.auth_key,
            "content-type": "application/json"
        }

        response = requests.post(
            f"{self.base_url_v4}/sms/send",
            headers=headers,
            json=payload,
            timeout=30
        )

        result = response.json()

        if response.status_code == 200 and result.get("type") == "success":
            message_id = result.get("message_id", str(uuid.uuid4()))
            return self._success_response(clean_number, message_text, message_id, route)
        else:
            error_msg = result.get("message", result.get("msg", "Unknown error"))
            return self._error_response(clean_number, error_msg)

    def _send_flow_sms(self, clean_number: str, message_text: str, route: str, template_id: str = None) -> Dict:
        """
        Send SMS via MSG91 Flow API (v5)
        Requires pre-approved template/flow
        """
        flow_id = template_id or self.template_id or ""

        payload = {
            "flow_id": flow_id,
            "sender": self.sender_id,
            "mobiles": clean_number,
            "route": route,
        }

        # For flow API, the message content is defined in the template
        # We pass variables if needed
        if message_text:
            payload["VAR1"] = message_text

        # Remove empty flow_id for promotional
        if route == "1" and not flow_id:
            payload.pop("flow_id", None)

        headers = {
            "authkey": self.auth_key,
            "content-type": "application/json"
        }

        response = requests.post(
            f"{self.base_url_v5}/flow/",
            headers=headers,
            json=payload,
            timeout=30
        )

        result = response.json()

        if response.status_code == 200 and result.get("type") == "success":
            message_id = result.get("request_id", str(uuid.uuid4()))
            return self._success_response(clean_number, message_text, message_id, route)
        else:
            error_msg = result.get("message", "Unknown error")
            return self._error_response(clean_number, error_msg)

    def send_bulk_sms(
        self,
        numbers: List[str],
        message_text: str,
        route: str = "4",
        template_id: str = None,
        country_code: str = "91"
    ) -> Dict:
        """
        Send bulk SMS to multiple numbers

        Args:
            numbers: List of mobile numbers
            message_text: SMS content
            route: Route type
            template_id: Template ID for transactional
            country_code: Country code

        Returns:
            Dict with overall results
        """
        results = []
        for number in numbers:
            result = self.send_sms(number, message_text, route, template_id, country_code)
            results.append(result)

        successful = sum(1 for r in results if r['success'])

        return {
            'success': successful > 0,
            'total': len(numbers),
            'successful': successful,
            'failed': len(numbers) - successful,
            'results': results,
            'provider': 'msg91'
        }

    def send_sms_with_unicode(self, to_number: str, message_text: str, route: str = "4") -> Dict:
        """
        Send Unicode SMS (supports Hindi, Telugu, etc.)
        Uses Simple SMS API with unicode flag
        """
        if self.simulate:
            return self._simulate_send(to_number, message_text, route)

        try:
            clean_number = self._clean_phone_number(to_number)

            payload = {
                "sender": self.sender_id,
                "mobiles": clean_number,
                "route": route,
                "sms": message_text,
                "unicode": 1
            }

            headers = {
                "authkey": self.auth_key,
                "content-type": "application/json"
            }

            response = requests.post(
                f"{self.base_url_v4}/sms/send",
                headers=headers,
                json=payload,
                timeout=30
            )

            result = response.json()

            if response.status_code == 200 and result.get("type") == "success":
                message_id = result.get("message_id", str(uuid.uuid4()))
                return self._success_response(clean_number, message_text, message_id, route)
            else:
                return self._error_response(clean_number, result.get("message", "Unknown error"))

        except Exception as e:
            return self._error_response(to_number, str(e))

    def get_balance(self) -> Dict:
        """Check account balance"""
        if self.simulate:
            return {
                'success': True,
                'balance': 'SIMULATED',
                'currency': 'INR',
                'sms_remaining': 9999,
                'mode': 'simulation'
            }

        try:
            headers = {"authkey": self.auth_key}
            response = requests.get(
                f"{self.base_url_v4}/balance",
                headers=headers,
                timeout=10
            )
            return response.json()
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def get_delivery_report(self, message_id: str) -> Dict:
        """Get delivery report for a message"""
        if self.simulate:
            return {
                'success': True,
                'message_id': message_id,
                'status': 'delivered',
                'delivery_time': datetime.now().isoformat(),
                'mode': 'simulation'
            }

        try:
            headers = {"authkey": self.auth_key}
            params = {"request_id": message_id}
            response = requests.get(
                f"{self.base_url_v5}/reports",
                headers=headers,
                params=params,
                timeout=10
            )
            return response.json()
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def get_sent_history(self, to_number: str = None) -> List[Dict]:
        """Get local sent message history"""
        if to_number:
            return [m for m in self.sent_messages if m['to_number'] == to_number]
        return self.sent_messages

    def get_simulation_log(self) -> List[Dict]:
        """Get simulation log entries"""
        return self.simulation_log

    def _clean_phone_number(self, number: str, country_code: str = "91") -> str:
        """Clean and format phone number"""
        clean = number.replace("+", "").replace(" ", "").replace("-", "").replace("(", "").replace(")", "")

        # If it doesn't start with country code, prepend it
        if not clean.startswith(country_code):
            if clean.startswith("0"):
                clean = country_code + clean[1:]
            else:
                clean = country_code + clean

        return clean

    def _simulate_send(self, clean_number: str, message_text: str, route: str) -> Dict:
        """Simulate sending SMS (for testing without credentials)"""
        time.sleep(0.3)  # Simulate network delay
        message_id = f"MSG91_SIM_{uuid.uuid4().hex[:12].upper()}"

        record = {
            'to_number': clean_number,
            'message': message_text,
            'message_id': message_id,
            'status': 'sent',
            'timestamp': datetime.now().isoformat(),
            'route': route,
            'mode': 'simulation'
        }
        self.sent_messages.append(record)
        self.simulation_log.append({
            **record,
            'event': 'sms_sent_simulated'
        })

        logger.info(f"[SIMULATION] SMS sent to {clean_number}: {message_text[:50]}...")

        return {
            'success': True,
            'message_id': message_id,
            'status': 'sent',
            'timestamp': datetime.now().isoformat(),
            'provider': 'msg91',
            'mode': 'simulation',
            'note': 'This is a simulated SMS. Configure MSG91 auth_key for real sending.'
        }

    def _success_response(self, number: str, message: str, message_id: str, route: str) -> Dict:
        """Build success response"""
        record = {
            'to_number': number,
            'message': message,
            'message_id': message_id,
            'status': 'sent',
            'timestamp': datetime.now().isoformat(),
            'route': route
        }
        self.sent_messages.append(record)

        return {
            'success': True,
            'message_id': message_id,
            'status': 'sent',
            'timestamp': datetime.now().isoformat(),
            'provider': 'msg91'
        }

    def _error_response(self, number: str, error: str) -> Dict:
        """Build error response"""
        return {
            'success': False,
            'error': error,
            'to_number': number,
            'provider': 'msg91'
        }


# Example usage and testing
if __name__ == "__main__":
    # Test in simulation mode
    gateway = MSG91Gateway(simulate=True)
    result = gateway.send_sms("919876543210", "Test message from Lumina Board - सिमुलेशन मोड")
    print(json.dumps(result, indent=2))

    # Test bulk
    bulk = gateway.send_bulk_sms(
        ["919876543210", "919876543211"],
        "Bulk test message"
    )
    print(json.dumps(bulk, indent=2))

    # Check balance
    balance = gateway.get_balance()
    print(json.dumps(balance, indent=2))

    # View simulation log
    print("\nSimulation Log:")
    for entry in gateway.get_simulation_log():
        print(f"  - {entry['event']}: {entry['to_number']}")