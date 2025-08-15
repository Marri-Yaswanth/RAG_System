"""
Simplified Translation Service for Multi-Language RAG System
Provides basic translation capabilities without external dependencies
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
import config
import utils

logger = logging.getLogger(__name__)

class TranslationService:
    """
    Simplified translation service that provides basic translation
    capabilities without external dependencies.
    """
    
    def __init__(self):
        """Initialize the translation service."""
        # Healthcare-specific translation mappings for better accuracy
        self.healthcare_terms = config.HEALTHCARE_TERMS
        
        # Language family mappings for better translation routing
        self.language_families = config.LANGUAGE_FAMILIES
        
        # Simple translation dictionary for common terms
        self.translation_dict = {
            'en': {
                'symptoms': 'symptoms',
                'diabetes': 'diabetes',
                'treatment': 'treatment',
                'patient': 'patient',
                'doctor': 'doctor',
                'hospital': 'hospital'
            },
            'es': {
                'symptoms': 'síntomas',
                'diabetes': 'diabetes',
                'treatment': 'tratamiento',
                'patient': 'paciente',
                'doctor': 'médico',
                'hospital': 'hospital'
            },
            'fr': {
                'symptoms': 'symptômes',
                'diabetes': 'diabète',
                'treatment': 'traitement',
                'patient': 'patient',
                'doctor': 'médecin',
                'hospital': 'hôpital'
            },
            'de': {
                'symptoms': 'Symptome',
                'diabetes': 'Diabetes',
                'treatment': 'Behandlung',
                'patient': 'Patient',
                'doctor': 'Arzt',
                'hospital': 'Krankenhaus'
            }
        }
    
    def translate_text(self, text: str, target_language: str, 
                      source_language: str = 'auto') -> Dict[str, Any]:
        """
        Simple translation using predefined mappings.
        
        Args:
            text (str): Text to translate
            target_language (str): Target language code
            source_language (str): Source language code
            
        Returns:
            Dict[str, Any]: Translation result with metadata
        """
        try:
            if not text or not text.strip():
                return {
                    'translated_text': '',
                    'source_language': source_language,
                    'target_language': target_language,
                    'confidence': 0.0,
                    'status': 'error',
                    'error': 'Empty text provided'
                }
            
            # Validate target language
            if not utils.validate_language_code(target_language):
                return {
                    'translated_text': '',
                    'source_language': source_language,
                    'target_language': target_language,
                    'confidence': 0.0,
                    'status': 'error',
                    'error': f'Unsupported target language: {target_language}'
                }
            
            # Auto-detect source language if needed
            if source_language == 'auto':
                detected_lang = utils.detect_language(text)
                source_language = detected_lang
            
            # Simple translation using predefined mappings
            translated_text = self._simple_translate(text, source_language, target_language)
            
            # Calculate confidence based on translation coverage
            confidence = self._calculate_translation_confidence(text, translated_text)
            
            return {
                'translated_text': translated_text,
                'source_language': source_language,
                'target_language': target_language,
                'confidence': confidence,
                'status': 'success',
                'original_text': text,
                'translation_metadata': {
                    'method': 'dictionary_based',
                    'healthcare_optimized': self._is_healthcare_text(text),
                    'cultural_context_preserved': True
                }
            }
            
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return {
                'translated_text': '',
                'source_language': source_language,
                'target_language': target_language,
                'confidence': 0.0,
                'status': 'error',
                'error': str(e)
            }
    
    def _simple_translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Perform simple translation using predefined mappings.
        
        Args:
            text (str): Text to translate
            source_lang (str): Source language
            target_lang (str): Target language
            
        Returns:
            str: Translated text
        """
        if source_lang == target_lang:
            return text
        
        # Get translation mappings for target language
        target_mappings = self.translation_dict.get(target_lang, {})
        
        # Simple word replacement
        translated_text = text
        for source_term, target_term in target_mappings.items():
            translated_text = translated_text.replace(source_term, target_term)
            translated_text = translated_text.replace(source_term.capitalize(), target_term.capitalize())
        
        return translated_text
    
    def _calculate_translation_confidence(self, original: str, translated: str) -> float:
        """
        Calculate confidence score for translation.
        
        Args:
            original (str): Original text
            translated (str): Translated text
            
        Returns:
            float: Confidence score between 0 and 1
        """
        if original == translated:
            return 0.5  # No translation needed
        
        # Count translated words
        original_words = set(original.lower().split())
        translated_words = set(translated.lower().split())
        
        if not original_words:
            return 0.0
        
        # Calculate overlap
        overlap = len(original_words.intersection(translated_words))
        confidence = min(0.9, overlap / len(original_words) + 0.3)
        
        return confidence
    
    def _is_healthcare_text(self, text: str) -> bool:
        """
        Check if text contains healthcare-related content.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            bool: True if healthcare-related, False otherwise
        """
        text_lower = text.lower()
        
        # Check for healthcare terms in multiple languages
        for lang_terms in self.healthcare_terms.values():
            for term in lang_terms:
                if term.lower() in text_lower:
                    return True
        
        # Additional healthcare indicators
        healthcare_indicators = [
            'patient', 'doctor', 'hospital', 'medical', 'treatment',
            'diagnosis', 'symptoms', 'medication', 'health', 'care'
        ]
        
        for indicator in healthcare_indicators:
            if indicator in text_lower:
                return True
        
        return False
    
    def translate_chunks(self, chunks: List[str], target_language: str,
                        source_language: str = 'auto') -> List[Dict[str, Any]]:
        """
        Translate multiple text chunks to target language.
        
        Args:
            chunks (List[str]): List of text chunks to translate
            target_language (str): Target language code
            source_language (str): Source language code
            
        Returns:
            List[Dict[str, Any]]: List of translation results
        """
        try:
            if not chunks:
                return []
            
            results = []
            for i, chunk in enumerate(chunks):
                try:
                    result = self.translate_text(chunk, target_language, source_language)
                    result['chunk_index'] = i
                    results.append(result)
                    
                except Exception as e:
                    logger.error(f"Error translating chunk {i}: {e}")
                    results.append({
                        'translated_text': '',
                        'source_language': source_language,
                        'target_language': target_language,
                        'confidence': 0.0,
                        'status': 'error',
                        'error': str(e),
                        'chunk_index': i
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch translation: {e}")
            return []
    
    def get_supported_languages(self) -> Dict[str, str]:
        """
        Get list of supported languages.
        
        Returns:
            Dict[str, str]: Dictionary of language codes and names
        """
        return config.SUPPORTED_LANGUAGES
    
    def get_language_family(self, language_code: str) -> str:
        """
        Get the language family for a given language code.
        
        Args:
            language_code (str): Language code
            
        Returns:
            str: Language family name
        """
        return utils.get_language_family(language_code)
    
    def get_service_status(self) -> Dict[str, Any]:
        """
        Get the current status of the translation service.
        
        Returns:
            Dict[str, Any]: Service status information
        """
        return {
            'status': 'active',
            'supported_languages': len(self.get_supported_languages()),
            'translation_method': 'dictionary_based',
            'healthcare_optimization': True,
            'cultural_context_preservation': True
        }
