"""
Utility functions for the Multi-Language RAG System
"""
import re
import hashlib
import logging
from typing import List, Dict, Tuple, Optional
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import config

# Set seed for consistent language detection
DetectorFactory.seed = 0

# Configure logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL), format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

def detect_language(text: str) -> str:
    """
    Detect the language of the given text.
    
    Args:
        text (str): Input text to detect language from
        
    Returns:
        str: Language code (e.g., 'en', 'es', 'fr')
    """
    try:
        # Clean text for better detection
        cleaned_text = re.sub(r'[^\w\s]', '', text.strip())
        if len(cleaned_text) < 10:
            return config.DEFAULT_SOURCE_LANGUAGE
        
        detected_lang = detect(cleaned_text)
        
        # Validate detected language
        if detected_lang in config.SUPPORTED_LANGUAGES:
            return detected_lang
        else:
            logger.warning(f"Detected unsupported language: {detected_lang}, defaulting to English")
            return config.DEFAULT_SOURCE_LANGUAGE
            
    except LangDetectException as e:
        logger.error(f"Language detection failed: {e}")
        return config.DEFAULT_SOURCE_LANGUAGE
    except Exception as e:
        logger.error(f"Unexpected error in language detection: {e}")
        return config.DEFAULT_SOURCE_LANGUAGE

def get_language_family(lang_code: str) -> str:
    """
    Get the language family for a given language code.
    
    Args:
        lang_code (str): Language code
        
    Returns:
        str: Language family name
    """
    for family, languages in config.LANGUAGE_FAMILIES.items():
        if lang_code in languages:
            return family
    return 'default'

def clean_text(text: str, language: str = 'en') -> str:
    """
    Clean and normalize text based on language.
    
    Args:
        text (str): Input text to clean
        language (str): Language code for language-specific cleaning
        
    Returns:
        str: Cleaned text
    """
    if not text:
        return ""
    
    # Basic cleaning
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    
    # Language-specific cleaning
    if language in ['zh', 'ja', 'ko']:
        # For Asian languages, preserve characters and basic punctuation
        text = re.sub(r'[^\w\s\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]', '', text)
    elif language == 'ar':
        # For Arabic, preserve Arabic characters and basic punctuation
        text = re.sub(r'[^\w\s\u0600-\u06ff]', '', text)
    else:
        # For other languages, remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:()]', '', text)
    
    return text

def create_chunks(text: str, language: str = 'en') -> List[str]:
    """
    Create text chunks based on language-specific strategies.
    
    Args:
        text (str): Input text to chunk
        language (str): Language code for chunking strategy
        
    Returns:
        List[str]: List of text chunks
    """
    strategy = get_chunking_strategy(language)
    chunk_size = strategy['chunk_size']
    chunk_overlap = strategy['chunk_overlap']
    separators = strategy['separators']
    
    chunks = []
    current_chunk = ""
    
    # Split text into sentences/paragraphs
    sentences = split_text_by_separators(text, separators)
    
    for sentence in sentences:
        # If adding this sentence would exceed chunk size
        if len(current_chunk + sentence) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # Start new chunk with overlap
            overlap_text = current_chunk[-chunk_overlap:] if chunk_overlap > 0 else ""
            current_chunk = overlap_text + sentence
        else:
            current_chunk += sentence
    
    # Add the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks

def split_text_by_separators(text: str, separators: List[str]) -> List[str]:
    """
    Split text using multiple separators in order of preference.
    
    Args:
        text (str): Input text
        separators (List[str]): List of separators in order of preference
        
    Returns:
        List[str]: List of text segments
    """
    segments = [text]
    
    for separator in separators:
        new_segments = []
        for segment in segments:
            if separator in segment:
                new_segments.extend(segment.split(separator))
            else:
                new_segments.append(segment)
        segments = new_segments
    
    # Clean up empty segments
    segments = [seg.strip() for seg in segments if seg.strip()]
    return segments

def get_chunking_strategy(language: str) -> Dict:
    """
    Get chunking strategy for a specific language.
    
    Args:
        language (str): Language code
        
    Returns:
        Dict: Chunking strategy configuration
    """
    family = get_language_family(language)
    
    if family == 'asian':
        return config.CHUNKING_STRATEGIES['asian']
    elif family == 'semitic':
        return config.CHUNKING_STRATEGIES['arabic']
    else:
        return config.CHUNKING_STRATEGIES['default']

def generate_document_id(content: str, filename: str = "") -> str:
    """
    Generate a unique document ID based on content and filename.
    
    Args:
        content (str): Document content
        filename (str): Original filename
        
    Returns:
        str: Unique document ID
    """
    # Create hash from content and filename
    hash_input = f"{content[:1000]}{filename}".encode('utf-8')
    return hashlib.md5(hash_input).hexdigest()

def extract_metadata(text: str, language: str, filename: str = "") -> Dict:
    """
    Extract metadata from text and language information.
    
    Args:
        text (str): Document text
        language (str): Detected language
        filename (str): Original filename
        
    Returns:
        Dict: Metadata dictionary
    """
    return {
        'language': language,
        'language_family': get_language_family(language),
        'filename': filename,
        'document_id': generate_document_id(text, filename),
        'text_length': len(text),
        'chunk_count': len(create_chunks(text, language)),
        'detection_confidence': 'high' if len(text) > 100 else 'low'
    }

def validate_language_code(lang_code: str) -> bool:
    """
    Validate if a language code is supported.
    
    Args:
        lang_code (str): Language code to validate
        
    Returns:
        bool: True if supported, False otherwise
    """
    return lang_code in config.SUPPORTED_LANGUAGES

def get_language_display_name(lang_code: str) -> str:
    """
    Get the display name for a language code.
    
    Args:
        lang_code (str): Language code
        
    Returns:
        str: Language display name
    """
    return config.SUPPORTED_LANGUAGES.get(lang_code, 'Unknown')

def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity between two text strings using basic metrics.
    
    Args:
        text1 (str): First text string
        text2 (str): Second text string
        
    Returns:
        float: Similarity score between 0 and 1
    """
    if not text1 or not text2:
        return 0.0
    
    # Convert to sets of words for Jaccard similarity
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 and not words2:
        return 1.0
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union)

def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe storage.
    
    Args:
        filename (str): Original filename
        
    Returns:
        str: Sanitized filename
    """
    # Remove or replace unsafe characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    sanitized = re.sub(r'\s+', '_', sanitized)
    return sanitized[:255]  # Limit length

def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes (int): Size in bytes
        
    Returns:
        str: Formatted size string
    """
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f}{size_names[i]}"
