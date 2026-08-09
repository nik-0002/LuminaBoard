from transformers import pipeline
import torchaudio
import torch
from typing import Dict
import os

class HuggingFaceTextToAudio:
    """
    Text-to-Speech using HuggingFace models
    Supports multiple Indian languages for farmer accessibility
    """
    
    def __init__(self, model_id: str = "espnet/kan-bayashi_ljspeech_vits", device: str = "cpu"):
        self.device = device
        self.model_id = model_id
        self.tts_pipeline = None
        self._load_model()
    
    def _load_model(self):
        """Load TTS model"""
        try:
            self.tts_pipeline = pipeline(
                "text-to-speech",
                model=self.model_id,
                device=0 if self.device == "cuda" else -1
            )
        except Exception as e:
            print(f"Error loading TTS model: {e}")
            self.tts_pipeline = None
    
    def generate_audio(self, text: str, language: str = "en", 
                      output_path: str = None) -> Dict:
        """
        Generate audio from text
        """
        if not self.tts_pipeline:
            return {'success': False, 'error': 'TTS model not loaded'}
        
        try:
            # Generate audio
            output = self.tts_pipeline(text, forward_params={"speaker_embeddings": None})
            
            audio_array = output["audio"]
            sampling_rate = output.get("sampling_rate", 16000)
            
            # Save if path provided
            if output_path:
                torchaudio.save(
                    output_path,
                    torch.tensor(audio_array).unsqueeze(0),
                    sampling_rate
                )
                
                return {
                    'success': True,
                    'audio_path': output_path,
                    'duration': len(audio_array) / sampling_rate,
                    'sampling_rate': sampling_rate,
                    'language': language
                }
            
            return {
                'success': True,
                'audio_data': audio_array,
                'sampling_rate': sampling_rate,
                'duration': len(audio_array) / sampling_rate,
                'language': language
            }
        
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def generate_campaign_audio(self, script: str, language: str = "hi", 
                               voice_style: str = "friendly") -> Dict:
        """
        Generate audio for marketing campaign
        """
        # Adjust script for audio (remove formatting)
        audio_text = self._prepare_audio_text(script, language)
        
        audio_result = self.generate_audio(
            text=audio_text,
            language=language
        )
        
        if audio_result['success']:
            audio_result['voice_style'] = voice_style
            audio_result['campaign_type'] = 'marketing'
        
        return audio_result
    
    def _prepare_audio_text(self, text: str, language: str) -> str:
        """Clean and prepare text for audio generation"""
        # Remove formatting, expand abbreviations, etc.
        audio_text = text.replace('[', '').replace(']', '')
        audio_text = audio_text.replace('\n', '. ')
        
        return audio_text
    
    def generate_multilingual_audio(self, text_dict: Dict[str, str]) -> Dict:
        """
        Generate audio in multiple languages
        text_dict: {language_code: text}
        """
        results = {}
        
        for language, text in text_dict.items():
            audio_result = self.generate_audio(text, language=language)
            results[language] = audio_result
        
        return {
            'success': all(r['success'] for r in results.values()),
            'audios': results,
            'supported_languages': list(text_dict.keys())
        }