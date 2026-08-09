"""
Lumina Board - Intelligent Message Routing Orchestrator
Routes messages across MSG91 SMS, WhatsApp (ADB/Business API), and Twilio
Supports simulation mode for testing without credentials
"""
import json
import logging
from typing import Dict, List, Optional
from datetime import datetime
from enum import Enum

logger = logging.getLogger("lumina.messaging.orchestrator")


class MessageChannel(Enum):
    WHATSAPP_ADB = "whatsapp_adb"
    WHATSAPP_API = "whatsapp_api"
    SMS_FAST2SMS = "sms_fast2sms"
    SMS_MSG91 = "sms_msg91"
    SMS_TWILIO = "sms_twilio"
    SMS_ADB = "sms_adb"
    VOICE = "voice"


class MessagingOrchestrator:
    """
    Intelligent message routing across multiple channels.
    
    Channels supported:
      - SMS_FAST2SMS: Fast2SMS API gateway (Free developer credits / India)
      - SMS_MSG91: MSG91 SMS gateway (India-optimized)
      - SMS_TWILIO: Twilio SMS gateway
      - WHATSAPP_ADB: WhatsApp via ADB (Android phone)
      - WHATSAPP_API: WhatsApp Business Cloud API
      - VOICE: Voice calls (future)
    """

    def __init__(
        self,
        msg91_gateway=None,
        whatsapp_channel=None,
        twilio_gateway=None,
        fast2sms_gateway=None,
        adb_controller=None,
        simulate: bool = True
    ):
        """
        Initialize orchestrator with available channels
        """
        self.msg91 = msg91_gateway
        self.whatsapp = whatsapp_channel
        self.twilio = twilio_gateway
        self.fast2sms = fast2sms_gateway
        self.adb = adb_controller
        self.simulate = simulate

        self.delivery_logs = []
        self.routing_rules = {}
        self.fallback_chain = [
            MessageChannel.SMS_FAST2SMS,
            MessageChannel.SMS_MSG91,
            MessageChannel.WHATSAPP_ADB,
            MessageChannel.WHATSAPP_API,
            MessageChannel.SMS_TWILIO,
            MessageChannel.SMS_ADB
        ]

        logger.info(f"Messaging Orchestrator initialized (simulate={simulate})")

    # ─── Public API ─────────────────────────────────────────────────────────────

    def route_campaign_message(
        self,
        farmer_context: Dict,
        message_content: Dict,
        campaign_id: str,
        preferred_channel: MessageChannel = None
    ) -> Dict:
        """
        Route message through optimal channel based on farmer profile.
        Falls back to alternative channels on failure.

        Args:
            farmer_context: Dict with farmer_id, phone_number, device_type, etc.
            message_content: Dict with text, media_urls, type
            campaign_id: Campaign identifier
            preferred_channel: Force a specific channel (optional)

        Returns:
            Dict with delivery result
        """
        # Select channel
        if preferred_channel:
            channel = preferred_channel
        else:
            channel = self._select_channel(farmer_context)

        # Attempt delivery with fallback
        result = self._deliver_with_fallback(
            channel=channel,
            farmer_context=farmer_context,
            message_content=message_content,
            campaign_id=campaign_id
        )

        # Log delivery
        self._log_delivery(
            campaign_id=campaign_id,
            farmer_id=farmer_context.get('farmer_id'),
            channel=result.get('channel_used', channel.value),
            status='sent' if result.get('success') else 'failed',
            message_type=message_content.get('type'),
            details=result
        )

        return result

    def send_sms(
        self,
        to_number: str,
        message_text: str,
        campaign_id: str = None,
        route: str = "4"
    ) -> Dict:
        """
        Send SMS directly via MSG91 (primary) or Twilio (fallback)

        Args:
            to_number: Recipient phone number
            message_text: SMS content
            campaign_id: Optional campaign identifier
            route: "4" transactional, "1" promotional

        Returns:
            Dict with delivery result
        """
        # Try Fast2SMS first (Free dev API / India)
        if self.fast2sms:
            result = self.fast2sms.send_sms(to_number, message_text)
            if result.get('success'):
                self._log_delivery(
                    campaign_id=campaign_id or 'direct_sms',
                    farmer_id=None,
                    channel=result.get('provider', 'sms_fast2sms'),
                    status='sent',
                    message_type='sms',
                    details=result
                )
                result['channel_used'] = result.get('provider', 'sms_fast2sms')
                return result

        # Try MSG91
        if self.msg91:
            result = self.msg91.send_sms(to_number, message_text, route)
            if result.get('success'):
                self._log_delivery(
                    campaign_id=campaign_id or 'direct_sms',
                    farmer_id=None,
                    channel='sms_msg91',
                    status='sent',
                    message_type='sms',
                    details=result
                )
                result['channel_used'] = 'sms_msg91'
                return result

        # Fallback to Twilio
        if self.twilio:
            result = self.twilio.send_sms(to_number, message_text)
            if result.get('success'):
                self._log_delivery(
                    campaign_id=campaign_id or 'direct_sms',
                    farmer_id=None,
                    channel='sms_twilio',
                    status='sent',
                    message_type='sms',
                    details=result
                )
                result['channel_used'] = 'sms_twilio'
                return result

        # Simulation fallback
        if self.simulate:
            sim_result = {
                'success': True,
                'message_id': f"F2S_SIM_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                'to_number': to_number,
                'status': 'sent',
                'timestamp': datetime.now().isoformat(),
                'provider': 'fast2sms_simulation',
                'channel_used': 'sms_fast2sms_simulated',
                'note': 'Simulated Fast2SMS send. Provide your API key in Dashboard Settings for live SMS.'
            }
            self._log_delivery(
                campaign_id=campaign_id or 'direct_sms',
                farmer_id=None,
                channel='sms_fast2sms_simulated',
                status='sent',
                message_type='sms',
                details=sim_result
            )
            return sim_result

        return {'success': False, 'error': 'No SMS channel available', 'channel_used': None}

    def send_whatsapp(
        self,
        to_number: str,
        message_text: str,
        campaign_id: str = None,
        media_url: str = None
    ) -> Dict:
        """
        Send WhatsApp message via best available channel

        Args:
            to_number: Recipient phone number
            message_text: Message content
            campaign_id: Optional campaign identifier
            media_url: Optional media attachment URL

        Returns:
            Dict with delivery result
        """
        # Try WhatsApp channel first (supports ADB, Business API, Simulation)
        if self.whatsapp:
            result = self.whatsapp.send_message(to_number, message_text, media_url)
            if result.get('success'):
                self._log_delivery(
                    campaign_id=campaign_id or 'direct_whatsapp',
                    farmer_id=None,
                    channel=f"whatsapp_{self.whatsapp.mode}",
                    status='sent',
                    message_type='whatsapp',
                    details=result
                )
                result['channel_used'] = f"whatsapp_{self.whatsapp.mode}"
                return result

        # Fallback to ADB controller (legacy)
        if self.adb:
            result = self.adb.send_whatsapp_message(to_number, message_text, media_url)
            if result.get('success'):
                self._log_delivery(
                    campaign_id=campaign_id or 'direct_whatsapp',
                    farmer_id=None,
                    channel='whatsapp_adb',
                    status='sent',
                    message_type='whatsapp',
                    details=result
                )
                result['channel_used'] = 'whatsapp_adb'
                return result

        # Simulation fallback
        if self.simulate:
            sim_result = {
                'success': True,
                'message_id': f"WA_SIM_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                'phone_number': to_number,
                'status': 'sent',
                'timestamp': datetime.now().isoformat(),
                'provider': 'simulation',
                'channel_used': 'whatsapp_simulated',
                'note': 'Simulated WhatsApp - configure credentials for real sending'
            }
            self._log_delivery(
                campaign_id=campaign_id or 'direct_whatsapp',
                farmer_id=None,
                channel='whatsapp_simulated',
                status='sent',
                message_type='whatsapp',
                details=sim_result
            )
            return sim_result

        return {'success': False, 'error': 'No WhatsApp channel available', 'channel_used': None}

    def send_bulk_sms(
        self,
        numbers: List[str],
        message_text: str,
        campaign_id: str = None,
        route: str = "4"
    ) -> Dict:
        """Send SMS to multiple recipients"""
        results = []
        for number in numbers:
            result = self.send_sms(number, message_text, campaign_id, route)
            results.append(result)

        successful = sum(1 for r in results if r.get('success', False))
        return {
            'success': successful > 0,
            'total': len(numbers),
            'successful': successful,
            'failed': len(numbers) - successful,
            'results': results,
            'campaign_id': campaign_id
        }

    def send_bulk_whatsapp(
        self,
        numbers: List[str],
        message_text: str,
        campaign_id: str = None,
        media_url: str = None
    ) -> Dict:
        """Send WhatsApp message to multiple recipients"""
        results = []
        for number in numbers:
            result = self.send_whatsapp(number, message_text, campaign_id, media_url)
            results.append(result)

        successful = sum(1 for r in results if r.get('success', False))
        return {
            'success': successful > 0,
            'total': len(numbers),
            'successful': successful,
            'failed': len(numbers) - successful,
            'results': results,
            'campaign_id': campaign_id
        }

    def get_delivery_status(self, campaign_id: str) -> Dict:
        """Get delivery status for a campaign"""
        logs = [l for l in self.delivery_logs if l['campaign_id'] == campaign_id]

        successful = len([l for l in logs if l['status'] == 'sent'])
        total = len(logs)

        return {
            'campaign_id': campaign_id,
            'total_messages': total,
            'successful_delivery': successful,
            'failed': total - successful,
            'delivery_rate': round((successful / total * 100), 2) if total > 0 else 0,
            'by_channel': self._group_by_channel(logs),
            'timestamp': datetime.now().isoformat()
        }

    def get_all_delivery_logs(self, limit: int = 100) -> List[Dict]:
        """Get recent delivery logs"""
        return self.delivery_logs[-limit:]

    def check_channel_status(self) -> Dict:
        """Check status of all configured channels"""
        status = {}

        if self.fast2sms:
            status['sms_fast2sms'] = {
                'configured': True,
                'simulating': self.fast2sms.simulate,
                'messages_sent': len(self.fast2sms.sent_messages)
            }
        else:
            status['sms_fast2sms'] = {'configured': False}

        if self.msg91:
            balance = self.msg91.get_balance()
            status['sms_msg91'] = {
                'configured': True,
                'simulating': self.msg91.simulate,
                'balance': balance.get('balance', 'unknown'),
                'messages_sent': len(self.msg91.sent_messages)
            }
        else:
            status['sms_msg91'] = {'configured': False}

        if self.whatsapp:
            conn = self.whatsapp.check_connection()
            status['whatsapp'] = {
                'configured': True,
                'mode': self.whatsapp.mode,
                'connected': conn.get('connected', False),
                'messages_sent': len(self.whatsapp.sent_history)
            }
        else:
            status['whatsapp'] = {'configured': False}

        if self.twilio:
            status['sms_twilio'] = {'configured': True}
        else:
            status['sms_twilio'] = {'configured': False}

        if self.adb:
            status['whatsapp_adb_legacy'] = {'configured': True}
        else:
            status['whatsapp_adb_legacy'] = {'configured': False}

        status['simulation_mode'] = self.simulate
        return status

    # ─── Channel Selection ──────────────────────────────────────────────────────

    def _select_channel(self, farmer_context: Dict) -> MessageChannel:
        """
        Select optimal channel based on farmer characteristics.
        
        Priority: Device type > Connectivity > Time of day > Language support
        
        Decision matrix:
          - Smartphone + good connectivity → WhatsApp
          - Feature phone or low connectivity → SMS (Fast2SMS / MSG91)
          - Business hours → WhatsApp preferred
          - Non-business hours → SMS (less intrusive)
        """
        device_type = farmer_context.get('device_type', 'smartphone')
        connectivity = farmer_context.get('connectivity_level', 'medium')
        hour = datetime.now().hour

        # Smartphone with good connectivity → WhatsApp
        if device_type == 'smartphone' and connectivity in ['medium', 'high']:
            if self.whatsapp or self.adb:
                return MessageChannel.WHATSAPP_ADB
            return MessageChannel.SMS_FAST2SMS

        # Feature phone or low connectivity → SMS
        if device_type == 'feature_phone' or connectivity == 'low':
            return MessageChannel.SMS_FAST2SMS

        # Business hours (8 AM - 8 PM) → WhatsApp
        if 8 <= hour <= 20:
            if self.whatsapp or self.adb:
                return MessageChannel.WHATSAPP_ADB
            return MessageChannel.SMS_FAST2SMS

        # Default: Fast2SMS (India-optimized)
        return MessageChannel.SMS_FAST2SMS

    def _deliver_with_fallback(
        self,
        channel: MessageChannel,
        farmer_context: Dict,
        message_content: Dict,
        campaign_id: str
    ) -> Dict:
        """
        Attempt delivery on selected channel, fallback to alternatives on failure
        """
        phone_number = farmer_context.get('phone_number')
        if not phone_number:
            return {'success': False, 'error': 'No phone number provided'}

        # Try primary channel
        result = self._send_via_channel(channel, farmer_context, message_content, campaign_id)
        if result.get('success'):
            result['channel_used'] = channel.value
            return result

        # Try fallback channels
        for fallback in self.fallback_chain:
            if fallback == channel:
                continue  # Skip primary (already failed)

            logger.info(f"Falling back from {channel.value} to {fallback.value}")
            result = self._send_via_channel(fallback, farmer_context, message_content, campaign_id)
            if result.get('success'):
                result['channel_used'] = fallback.value
                result['fallback_from'] = channel.value
                return result

        # All channels failed
        return {
            'success': False,
            'error': 'All channels failed',
            'channel_used': channel.value,
            'phone_number': phone_number
        }

    def _send_via_channel(
        self,
        channel: MessageChannel,
        farmer_context: Dict,
        message_content: Dict,
        campaign_id: str
    ) -> Dict:
        """Execute sending via selected channel"""
        phone_number = farmer_context.get('phone_number')
        message_text = message_content.get('text', '')
        media_url = message_content.get('media_url') or message_content.get('media_urls', [None])[0]

        # Fast2SMS SMS
        if channel == MessageChannel.SMS_FAST2SMS:
            if self.fast2sms:
                return self.fast2sms.send_sms(
                    to_number=phone_number,
                    message_text=message_text
                )
            elif self.simulate:
                return {
                    'success': True,
                    'message_id': f"F2S_SIM_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    'status': 'sent',
                    'provider': 'fast2sms_simulation',
                    'note': 'Simulated Fast2SMS via orchestrator'
                }
            return {'success': False, 'error': 'Fast2SMS not configured'}

        # MSG91 SMS
        elif channel == MessageChannel.SMS_MSG91:
            if self.msg91:
                return self.msg91.send_sms(
                    to_number=phone_number,
                    message_text=message_text,
                    route="4"  # Transactional
                )
            elif self.simulate:
                return {
                    'success': True,
                    'message_id': f"SIM_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    'status': 'sent',
                    'provider': 'simulation',
                    'note': 'Simulated via orchestrator'
                }
            return {'success': False, 'error': 'MSG91 not configured'}

        # WhatsApp via ADB
        elif channel == MessageChannel.WHATSAPP_ADB:
            if self.whatsapp and self.whatsapp.mode in ('adb', 'simulate'):
                return self.whatsapp.send_message(
                    to_number=phone_number,
                    message_text=message_text,
                    media_url=media_url
                )
            elif self.adb:
                return self.adb.send_whatsapp_message(
                    phone_number=phone_number,
                    message=message_text,
                    media_path=media_url
                )
            elif self.simulate:
                return {
                    'success': True,
                    'message_id': f"WA_SIM_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    'status': 'sent',
                    'provider': 'simulation',
                    'note': 'Simulated WhatsApp via orchestrator'
                }
            return {'success': False, 'error': 'WhatsApp ADB not configured'}

        # WhatsApp via Business API
        elif channel == MessageChannel.WHATSAPP_API:
            if self.whatsapp and self.whatsapp.mode == 'business_api':
                return self.whatsapp.send_message(
                    to_number=phone_number,
                    message_text=message_text,
                    media_url=media_url
                )
            return {'success': False, 'error': 'WhatsApp Business API not configured'}

        # Twilio SMS
        elif channel == MessageChannel.SMS_TWILIO:
            if self.twilio:
                return self.twilio.send_sms(
                    to_number=phone_number,
                    message_text=message_text
                )
            return {'success': False, 'error': 'Twilio not configured'}

        # Voice (future)
        elif channel == MessageChannel.VOICE:
            return {'success': False, 'error': 'Voice channel not yet implemented'}

        return {'success': False, 'error': f'Unknown channel: {channel}'}

    # ─── Logging & Analytics ────────────────────────────────────────────────────

    def _log_delivery(
        self,
        campaign_id: str,
        farmer_id: str,
        channel: str,
        status: str,
        message_type: str,
        details: Dict = None
    ):
        """Log delivery attempt"""
        log_entry = {
            'campaign_id': campaign_id,
            'farmer_id': farmer_id,
            'channel': channel,
            'status': status,
            'message_type': message_type,
            'timestamp': datetime.now().isoformat(),
            'details': details or {}
        }
        self.delivery_logs.append(log_entry)
        logger.info(f"Delivery logged: campaign={campaign_id}, channel={channel}, status={status}")

    def _group_by_channel(self, logs: List[Dict]) -> Dict:
        """Group delivery logs by channel"""
        grouped = {}
        for log in logs:
            channel = log['channel']
            if channel not in grouped:
                grouped[channel] = {'sent': 0, 'failed': 0}

            if log['status'] == 'sent':
                grouped[channel]['sent'] += 1
            else:
                grouped[channel]['failed'] += 1

        return grouped


# Example usage
if __name__ == "__main__":
    from messaging.channels.msg91_gateway import MSG91Gateway
    from messaging.channels.adb_whatsapp import WhatsAppChannel

    # Initialize with simulation
    msg91 = MSG91Gateway(simulate=True)
    whatsapp = WhatsAppChannel(mode="simulate")

    orchestrator = MessagingOrchestrator(
        msg91_gateway=msg91,
        whatsapp_channel=whatsapp,
        simulate=True
    )

    # Test SMS
    sms_result = orchestrator.send_sms("919876543210", "Test SMS from orchestrator")
    print("SMS Result:", json.dumps(sms_result, indent=2))

    # Test WhatsApp
    wa_result = orchestrator.send_whatsapp("919876543210", "Test WhatsApp from orchestrator")
    print("\nWhatsApp Result:", json.dumps(wa_result, indent=2))

    # Test campaign routing
    campaign_result = orchestrator.route_campaign_message(
        farmer_context={
            'farmer_id': 'GRW_00001',
            'phone_number': '919876543210',
            'device_type': 'smartphone',
            'connectivity_level': 'high',
            'language': 'Telugu'
        },
        message_content={
            'text': 'నమస్కారం రైతు అన్నా! Lumina Board కొత్త ఉత్పత్తి అందుబాటులో ఉంది.',
            'type': 'product_launch'
        },
        campaign_id='CAMP_TEST_001'
    )
    print("\nCampaign Result:", json.dumps(campaign_result, indent=2))

    # Check status
    status = orchestrator.check_channel_status()
    print("\nChannel Status:", json.dumps(status, indent=2))