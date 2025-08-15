"""
Configuration file for the Multi-Language RAG System
"""
import os
from typing import Dict, List

# Supported languages with their codes and display names
SUPPORTED_LANGUAGES = {
    'en': 'English',
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'it': 'Italian',
    'pt': 'Portuguese',
    'zh': 'Chinese',
    'ja': 'Japanese',
    'ko': 'Korean',
    'ar': 'Arabic',
    'hi': 'Hindi',
    'ru': 'Russian'
}

# Language families for better translation routing
LANGUAGE_FAMILIES = {
    'romance': ['es', 'fr', 'it', 'pt'],
    'germanic': ['en', 'de'],
    'slavic': ['ru'],
    'asian': ['zh', 'ja', 'ko'],
    'semitic': ['ar'],
    'indo-aryan': ['hi']
}

# Model configuration
EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL_NAME', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', 1000))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', 200))

# Vector database configuration
CHROMA_PERSIST_DIRECTORY = os.getenv('CHROMA_PERSIST_DIRECTORY', './chroma_db')
COLLECTION_NAME = 'multilingual_documents'

# RAG configuration
TOP_K_RESULTS = int(os.getenv('TOP_K_RESULTS', 5))
SIMILARITY_THRESHOLD = float(os.getenv('SIMILARITY_THRESHOLD', 0.3))

# Translation configuration
DEFAULT_SOURCE_LANGUAGE = os.getenv('DEFAULT_SOURCE_LANGUAGE', 'en')
DEFAULT_TARGET_LANGUAGE = os.getenv('DEFAULT_TARGET_LANGUAGE', 'en')

# Healthcare domain specific terms (for better chunking and retrieval)
HEALTHCARE_TERMS = {
    'en': ['patient', 'diagnosis', 'treatment', 'medication', 'symptoms', 'doctor', 'hospital'],
    'es': ['paciente', 'diagnóstico', 'tratamiento', 'medicamento', 'síntomas', 'médico', 'hospital'],
    'fr': ['patient', 'diagnostic', 'traitement', 'médicament', 'symptômes', 'médecin', 'hôpital'],
    'de': ['Patient', 'Diagnose', 'Behandlung', 'Medikament', 'Symptome', 'Arzt', 'Krankenhaus'],
    'it': ['paziente', 'diagnosi', 'trattamento', 'medicamento', 'sintomi', 'medico', 'ospedale'],
    'pt': ['paciente', 'diagnóstico', 'tratamento', 'medicamento', 'sintomas', 'médico', 'hospital'],
    'zh': ['患者', '诊断', '治疗', '药物', '症状', '医生', '医院'],
    'ja': ['患者', '診断', '治療', '薬', '症状', '医師', '病院'],
    'ko': ['환자', '진단', '치료', '약물', '증상', '의사', '병원'],
    'ar': ['مريض', 'تشخيص', 'علاج', 'دواء', 'أعراض', 'طبيب', 'مستشفى'],
    'hi': ['रोगी', 'निदान', 'उपचार', 'दवा', 'लक्षण', 'डॉक्टर', 'अस्पताल'],
    'ru': ['пациент', 'диагноз', 'лечение', 'лекарство', 'симптомы', 'врач', 'больница']
}

# File type support
SUPPORTED_FILE_TYPES = {
    'text': ['.txt', '.md'],
    'document': ['.pdf', '.docx', '.doc'],
    'data': ['.csv', '.json', '.xml']
}

# Chunking strategies for different languages
CHUNKING_STRATEGIES = {
    'default': {
        'chunk_size': CHUNK_SIZE,
        'chunk_overlap': CHUNK_OVERLAP,
        'separators': ['\n\n', '\n', '. ', '! ', '? ', '; ', ', ']
    },
    'asian': {
        'chunk_size': CHUNK_SIZE // 2,  # Smaller chunks for character-based languages
        'chunk_overlap': CHUNK_OVERLAP // 2,
        'separators': ['。', '！', '？', '；', '，', '\n\n', '\n']
    },
    'arabic': {
        'chunk_size': CHUNK_SIZE,
        'chunk_overlap': CHUNK_OVERLAP,
        'separators': ['\n\n', '\n', '. ', '! ', '? ', '; ', '، ']
    }
}

# Evaluation metrics configuration
EVALUATION_METRICS = {
    'answer_relevancy': True,
    'context_relevancy': True,
    'faithfulness': True,
    'context_recall': True
}

# API rate limiting
RATE_LIMITS = {
    'translation_per_minute': 60,
    'embedding_per_minute': 100,
    'query_per_minute': 120
}

# Cache configuration
CACHE_TTL = 3600  # 1 hour
CACHE_MAX_SIZE = 1000

# Logging configuration
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
