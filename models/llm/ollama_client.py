import requests
import json
from typing import List, Dict
import time

class OllamaLLMClient:
    """
    Client for Ollama-hosted Qwen2.5 model
    Generates dynamic marketing content based on farmer context
    """
    
    def __init__(self, base_url="http://localhost:11434", model="qwen2.5"):
        self.base_url = base_url
        self.model = model
        self.api_endpoint = f"{base_url}/api/generate"
        self.embedding_endpoint = f"{base_url}/api/embeddings"
    
    def check_connection(self) -> bool:
        """Verify connection to Ollama service"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def generate_content(self, prompt: str, temperature=0.7, top_p=0.9, max_tokens=512) -> Dict:
        """
        Generate marketing content using Qwen2.5
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": temperature,
            "top_p": top_p,
            "num_predict": max_tokens,
            "stream": False
        }
        
        try:
            response = requests.post(self.api_endpoint, json=payload, timeout=60)
            if response.status_code == 200:
                result = response.json()
                return {
                    'success': True,
                    'content': result.get('response', ''),
                    'model': result.get('model'),
                    'eval_count': result.get('eval_count'),
                    'prompt_eval_count': result.get('prompt_eval_count')
                }
            else:
                return {'success': False, 'error': f"API returned {response.status_code}"}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_embeddings(self, text: str) -> Dict:
        """Get embeddings for RAG vector store"""
        payload = {
            "model": self.model,
            "prompt": text
        }
        
        try:
            response = requests.post(self.embedding_endpoint, json=payload, timeout=30)
            if response.status_code == 200:
                result = response.json()
                return {
                    'success': True,
                    'embedding': result.get('embedding'),
                    'embedding_size': len(result.get('embedding', []))
                }
            else:
                return {'success': False, 'error': f"API returned {response.status_code}"}
        except Exception as e:
            return {'success': False, 'error': str(e)}

class ContentGenerator:
    """
    Generates personalized marketing content using RAG + LLM
    """
    
    def __init__(self, llm_client: OllamaLLMClient, rag_retriever):
        self.llm = llm_client
        self.retriever = rag_retriever
    
    def generate_whatsapp_message(self, farmer_context: Dict) -> Dict:
        """
        Generate WhatsApp message tailored to farmer
        """
        # Retrieve relevant government data and best practices
        rag_context = self.retriever.retrieve_context(
            crop=farmer_context['crop'],
            region=farmer_context['region'],
            current_stage=farmer_context['growth_stage'],
            query_type='pest_management'
        )
        
        prompt = f"""Generate a concise, friendly WhatsApp message in {farmer_context.get('language', 'Hindi')} for an Indian farmer.

Farmer Profile:
- Crop: {farmer_context['crop']}
- Region: {farmer_context['region']}
- Farm size: {farmer_context.get('farm_size', 2.5)} acres
- Growth stage: {farmer_context['growth_stage']}
- Pest pressure: {farmer_context.get('pest_pressure', 'moderate')}
- Weather: {farmer_context.get('weather_condition', 'normal')}

Recommended Lumina Board Product: {farmer_context.get('product', 'suitable crop protection')}

Government Advisory: {rag_context.get('government_advisory', '')}

Best Practice: {rag_context.get('best_practice', '')}

Requirements:
- Keep it under 160 characters (fits SMS)
- Use simple, conversational language
- Include urgency if pest pressure is high
- Mention specific benefit for this crop
- End with clear call-to-action (inquiry/visit retailer)
- NO EMOJIS, professional tone with warmth

Message:"""
        
        result = self.llm.generate_content(prompt, temperature=0.6, max_tokens=200)
        
        return {
            'success': result['success'],
            'message': result.get('content', '').strip(),
            'rag_context_used': rag_context,
            'model_used': self.llm.model
        }
    
    def generate_video_script(self, farmer_context: Dict, duration_seconds=30) -> Dict:
        """
        Generate script for WhatsApp video message
        """
        rag_context = self.retriever.retrieve_context(
            crop=farmer_context['crop'],
            region=farmer_context['region'],
            current_stage=farmer_context['growth_stage'],
            query_type='video_content'
        )
        
        prompt = f"""Create a {duration_seconds}-second WhatsApp video script in {farmer_context.get('language', 'Hindi')} for Indian farmers.

Topic: {rag_context.get('topic', 'Crop protection')}
Target Farmer: {farmer_context['crop']} grower in {farmer_context['region']}
Lumina Board Product: {farmer_context.get('product', '')}

Key Points to Cover:
- Problem identification (pest/disease)
- Solution (product benefits)
- Application instructions
- Expected results

Script Format:
[VISUALS] | [AUDIO - FARMER-FRIENDLY LANGUAGE]
Keep sentences short. Use simple vocabulary.
Include: timestamp in brackets for visual changes."""
        
        result = self.llm.generate_content(prompt, temperature=0.7, max_tokens=400)
        
        return {
            'success': result['success'],
            'script': result.get('content', '').strip(),
            'duration': duration_seconds,
            'language': farmer_context.get('language', 'Hindi'),
            'rag_context': rag_context
        }
    
    def generate_voice_call_script(self, farmer_context: Dict) -> Dict:
        """
        Generate script for automated voice call or voice message
        """
        prompt = f"""Generate a voice call script for a field representative talking to a farmer about crop protection.

Farmer: {farmer_context['crop']} grower, {farmer_context['region']}
Language: {farmer_context.get('language', 'Hindi')}
Product: {farmer_context.get('product', '')}
Concern: {farmer_context.get('primary_concern', 'pest management')}

Script should:
- Start with friendly greeting
- Identify the problem relevant to their crop
- Explain the solution
- Provide application details
- End with next steps
- Be conversational and warm
- Duration: ~2-3 minutes
- Use phrases a farmer would understand

Format:
[Duration: X seconds] Speaker: CONTENT"""
        
        result = self.llm.generate_content(prompt, temperature=0.7, max_tokens=500)
        
        return {
            'success': result['success'],
            'script': result.get('content', '').strip(),
            'estimated_duration': '2-3 minutes',
            'language': farmer_context.get('language', 'Hindi')
        }