"""
Document Processor for Multi-Language RAG System
Handles document ingestion, language detection, chunking, and preprocessing
"""
import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import PyPDF2
import docx
import pandas as pd
import json
import xml.etree.ElementTree as ET

import config
import utils

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Handles processing of documents in multiple languages.
    Supports various file formats and implements language-specific chunking strategies.
    """
    
    def __init__(self):
        self.supported_extensions = {
            '.txt': self._process_text_file,
            '.md': self._process_text_file,
            '.pdf': self._process_pdf_file,
            '.docx': self._process_docx_file,
            '.doc': self._process_docx_file,
            '.csv': self._process_csv_file,
            '.json': self._process_json_file,
            '.xml': self._process_xml_file
        }
    
    def process_document(self, file_path: str) -> Dict[str, Any]:
        """
        Process a document and return structured data.
        
        Args:
            file_path (str): Path to the document file
            
        Returns:
            Dict[str, Any]: Processed document data with chunks and metadata
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Check file extension
            extension = file_path.suffix.lower()
            if extension not in self.supported_extensions:
                raise ValueError(f"Unsupported file type: {extension}")
            
            # Read and process the file
            content = self.supported_extensions[extension](file_path)
            
            # Detect language
            language = utils.detect_language(content)
            
            # Clean text based on language
            cleaned_content = utils.clean_text(content, language)
            
            # Create chunks
            chunks = utils.create_chunks(cleaned_content, language)
            
            # Extract metadata
            metadata = utils.extract_metadata(cleaned_content, language, file_path.name)
            
            # Add file-specific metadata
            metadata.update({
                'file_path': str(file_path),
                'file_size': file_path.stat().st_size,
                'file_extension': extension,
                'processed_chunks': len(chunks)
            })
            
            return {
                'content': cleaned_content,
                'chunks': chunks,
                'metadata': metadata,
                'language': language,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {e}")
            return {
                'content': '',
                'chunks': [],
                'metadata': {},
                'language': config.DEFAULT_SOURCE_LANGUAGE,
                'status': 'error',
                'error': str(e)
            }
    
    def _process_text_file(self, file_path: Path) -> str:
        """Process text files (.txt, .md)."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            # Try different encodings
            for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    with open(file_path, 'r', encoding=encoding) as file:
                        return file.read()
                except UnicodeDecodeError:
                    continue
            raise ValueError(f"Could not decode file with any supported encoding: {file_path}")
    
    def _process_pdf_file(self, file_path: Path) -> str:
        """Process PDF files."""
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except Exception as e:
            logger.error(f"Error processing PDF file {file_path}: {e}")
            raise
    
    def _process_docx_file(self, file_path: Path) -> str:
        """Process Word documents (.docx, .doc)."""
        try:
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            logger.error(f"Error processing Word document {file_path}: {e}")
            raise
    
    def _process_csv_file(self, file_path: Path) -> str:
        """Process CSV files."""
        try:
            df = pd.read_csv(file_path)
            # Convert DataFrame to text representation
            text = df.to_string(index=False)
            return text
        except Exception as e:
            logger.error(f"Error processing CSV file {file_path}: {e}")
            raise
    
    def _process_json_file(self, file_path: Path) -> str:
        """Process JSON files."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                # Convert JSON to formatted text
                return json.dumps(data, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error processing JSON file {file_path}: {e}")
            raise
    
    def _process_xml_file(self, file_path: Path) -> str:
        """Process XML files."""
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            # Convert XML to text representation
            return ET.tostring(root, encoding='unicode', method='xml')
        except Exception as e:
            logger.error(f"Error processing XML file {file_path}: {e}")
            raise
    
    def process_multiple_documents(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Process multiple documents in batch.
        
        Args:
            file_paths (List[str]): List of file paths to process
            
        Returns:
            List[Dict[str, Any]]: List of processed document data
        """
        results = []
        for file_path in file_paths:
            try:
                result = self.process_document(file_path)
                results.append(result)
                logger.info(f"Successfully processed: {file_path}")
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                results.append({
                    'content': '',
                    'chunks': [],
                    'metadata': {'file_path': file_path, 'error': str(e)},
                    'language': config.DEFAULT_SOURCE_LANGUAGE,
                    'status': 'error'
                })
        
        return results
    
    def validate_document(self, file_path: str) -> Tuple[bool, str]:
        """
        Validate if a document can be processed.
        
        Args:
            file_path (str): Path to the document file
            
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        try:
            file_path = Path(file_path)
            
            # Check if file exists
            if not file_path.exists():
                return False, "File does not exist"
            
            # Check file size (max 50MB)
            if file_path.stat().st_size > 50 * 1024 * 1024:
                return False, "File size exceeds 50MB limit"
            
            # Check file extension
            extension = file_path.suffix.lower()
            if extension not in self.supported_extensions:
                return False, f"Unsupported file type: {extension}"
            
            return True, "File is valid"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def get_processing_stats(self, processed_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about processed documents.
        
        Args:
            processed_docs (List[Dict[str, Any]]): List of processed documents
            
        Returns:
            Dict[str, Any]: Processing statistics
        """
        if not processed_docs:
            return {}
        
        total_docs = len(processed_docs)
        successful_docs = len([doc for doc in processed_docs if doc['status'] == 'success'])
        failed_docs = total_docs - successful_docs
        
        # Language distribution
        language_counts = {}
        total_chunks = 0
        total_content_length = 0
        
        for doc in processed_docs:
            if doc['status'] == 'success':
                lang = doc['language']
                language_counts[lang] = language_counts.get(lang, 0) + 1
                total_chunks += len(doc['chunks'])
                total_content_length += len(doc['content'])
        
        return {
            'total_documents': total_docs,
            'successful_documents': successful_docs,
            'failed_documents': failed_docs,
            'success_rate': successful_docs / total_docs if total_docs > 0 else 0,
            'language_distribution': language_counts,
            'total_chunks': total_chunks,
            'total_content_length': total_content_length,
            'average_chunks_per_doc': total_chunks / successful_docs if successful_docs > 0 else 0,
            'average_content_length': total_content_length / successful_docs if successful_docs > 0 else 0
        }
    
    def create_sample_documents(self) -> List[Dict[str, Any]]:
        """
        Create sample healthcare documents in multiple languages for testing.
        
        Returns:
            List[Dict[str, Any]]: List of sample documents
        """
        sample_docs = [
            {
                'language': 'en',
                'content': """
                Patient Care Guidelines for Diabetes Management
                
                Diabetes is a chronic condition that affects how your body turns food into energy. 
                With diabetes, your body either doesn't make enough insulin or can't use it as well as it should.
                
                Key Management Strategies:
                1. Regular blood glucose monitoring
                2. Balanced diet and portion control
                3. Regular physical activity
                4. Medication adherence
                5. Regular medical check-ups
                
                Symptoms to Watch For:
                - Increased thirst and frequent urination
                - Extreme hunger
                - Unexplained weight loss
                - Fatigue and irritability
                - Blurred vision
                
                Emergency Signs:
                If you experience severe symptoms like confusion, rapid breathing, or loss of consciousness, 
                seek immediate medical attention.
                """,
                'filename': 'diabetes_guidelines_en.txt'
            },
            {
                'language': 'es',
                'content': """
                Guías de Cuidado del Paciente para el Manejo de la Diabetes
                
                La diabetes es una condición crónica que afecta cómo su cuerpo convierte los alimentos en energía.
                Con la diabetes, su cuerpo no produce suficiente insulina o no puede usarla tan bien como debería.
                
                Estrategias Clave de Manejo:
                1. Monitoreo regular de glucosa en sangre
                2. Dieta balanceada y control de porciones
                3. Actividad física regular
                4. Adherencia a la medicación
                5. Chequeos médicos regulares
                
                Síntomas a Observar:
                - Aumento de sed y micción frecuente
                - Hambre extrema
                - Pérdida de peso inexplicable
                - Fatiga e irritabilidad
                - Visión borrosa
                
                Signos de Emergencia:
                Si experimenta síntomas severos como confusión, respiración rápida o pérdida de conciencia,
                busque atención médica inmediata.
                """,
                'filename': 'diabetes_guidelines_es.txt'
            },
            {
                'language': 'fr',
                'content': """
                Directives de Soins aux Patients pour la Gestion du Diabète
                
                Le diabète est une affection chronique qui affecte la façon dont votre corps transforme 
                les aliments en énergie. Avec le diabète, votre corps ne produit pas assez d'insuline 
                ou ne peut pas l'utiliser aussi bien qu'il le devrait.
                
                Stratégies Clés de Gestion:
                1. Surveillance régulière de la glycémie
                2. Régime alimentaire équilibré et contrôle des portions
                3. Activité physique régulière
                4. Adhésion aux médicaments
                5. Examens médicaux réguliers
                
                Symptômes à Surveiller:
                - Soif accrue et mictions fréquentes
                - Faim extrême
                - Perte de poids inexpliquée
                - Fatigue et irritabilité
                - Vision floue
                
                Signes d'Urgence:
                Si vous ressentez des symptômes graves comme la confusion, une respiration rapide 
                ou une perte de conscience, consultez immédiatement un médecin.
                """,
                'filename': 'diabetes_guidelines_fr.txt'
            }
        ]
        
        # Process sample documents
        processed_docs = []
        for doc in sample_docs:
            try:
                # Create temporary file content
                content = doc['content']
                language = doc['language']
                
                # Clean and chunk the content
                cleaned_content = utils.clean_text(content, language)
                chunks = utils.create_chunks(cleaned_content, language)
                
                # Extract metadata
                metadata = utils.extract_metadata(cleaned_content, language, doc['filename'])
                
                processed_docs.append({
                    'content': cleaned_content,
                    'chunks': chunks,
                    'metadata': metadata,
                    'language': language,
                    'status': 'success'
                })
                
            except Exception as e:
                logger.error(f"Error processing sample document {doc['filename']}: {e}")
        
        return processed_docs
