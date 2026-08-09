from twilio.rest import Client
from typing import Dict, List
from datetime import datetime

class TwilioMessagingGateway:
    """
    Twilio API gateway for WhatsApp and SMS messaging
    Hot-swappable alternative to ADB when API is preferred
    """
    
    def __init__(self, account_sid: str, auth_token: str, from_number: str):
        self.client = Client(account_sid, auth_token)
        self.from_number = from_number
        self.sent_messages = []
    
    def send_whatsapp_message(self, to_number: str, message_text: str, 
                             media_urls: List[str] = None) -> Dict:
        """
        Send WhatsApp message via Twilio API
        """
        try:
            if media_urls:
                # Send message with media
                message = self.client.messages.create(
                    from_=f"whatsapp:{self.from_number}",
                    to=f"whatsapp:{to_number}",
                    body=message_text,
                    media_url=media_urls
                )
            else:
                # Send text message
                message = self.client.messages.create(
                    from_=f"whatsapp:{self.from_number}",
                    to=f"whatsapp:{to_number}",
                    body=message_text
                )
            
            record = {
                'to_number': to_number,
                'message': message_text,
                'message_id': message.sid,
                'status': message.status,
                'timestamp': datetime.now().isoformat()
            }
            self.sent_messages.append(record)
            
            return {
                'success': True,
                'message_id': message.sid,
                'status': message.status,
                'timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'to_number': to_number
            }
    
    def send_sms(self, to_number: str, message_text: str) -> Dict:
        """Send SMS via Twilio"""
        try:
            message = self.client.messages.create(
                from_=self.from_number,
                to=to_number,
                body=message_text
            )
            
            record = {
                'to_number': to_number,
                'message': message_text,
                'message_id': message.sid,
                'status': message.status,
                'timestamp': datetime.now().isoformat()
            }
            self.sent_messages.append(record)
            
            return {
                'success': True,
                'message_id': message.sid,
                'status': message.status
            }
        
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }