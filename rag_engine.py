"""
RAG Engine for Multi-Language RAG System
Orchestrates document processing, embeddings, retrieval, and translation
"""
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

import config
import utils
from document_processor import DocumentProcessor
from embedding_engine import EmbeddingEngine
from translation_service import TranslationService
from vector_db import VectorDatabase

logger = logging.getLogger(__name__)

class RAGEngine:
    """
    Core RAG engine that orchestrates all components for multi-language
    document retrieval and question answering.
    """
    
    def __init__(self):
        """Initialize the RAG engine with all components."""
        self.document_processor = DocumentProcessor()
        self.embedding_engine = EmbeddingEngine()
        self.translation_service = TranslationService()
        self.vector_db = VectorDatabase()
        
        logger.info("RAG Engine initialized successfully")
    
    def process_and_index_documents(self, file_paths: List[str]) -> Dict[str, Any]:
        """
        Process documents and add them to the vector database.
        
        Args:
            file_paths (List[str]): List of file paths to process
            
        Returns:
            Dict[str, Any]: Processing results and statistics
        """
        try:
            start_time = time.time()
            
            # Process documents
            logger.info(f"Processing {len(file_paths)} documents...")
            processed_docs = self.document_processor.process_multiple_documents(file_paths)
            
            # Filter successful documents
            successful_docs = [doc for doc in processed_docs if doc['status'] == 'success']
            failed_docs = [doc for doc in processed_docs if doc['status'] == 'error']
            
            if not successful_docs:
                logger.warning("No documents were successfully processed")
                return {
                    'status': 'error',
                    'message': 'No documents were successfully processed',
                    'successful_docs': 0,
                    'failed_docs': len(failed_docs),
                    'processing_time': time.time() - start_time
                }
            
            # Create embeddings for all chunks
            all_chunks = []
            all_metadata = []
            
            for doc in successful_docs:
                chunks = doc.get('chunks', [])
                for i, chunk in enumerate(chunks):
                    all_chunks.append(chunk)
                    chunk_metadata = doc['metadata'].copy()
                    chunk_metadata.update({
                        'chunk_index': i,
                        'chunk_text': chunk[:200] + '...' if len(chunk) > 200 else chunk
                    })
                    all_metadata.append(chunk_metadata)
            
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(all_chunks)} chunks...")
            embeddings, enhanced_metadata = self.embedding_engine.encode_chunks(
                all_chunks, all_metadata
            )
            
            # Add to vector database
            logger.info("Adding embeddings to vector database...")
            chunk_ids = self.vector_db.add_chunks(all_chunks, embeddings, enhanced_metadata)
            
            # Get processing statistics
            processing_stats = self.document_processor.get_processing_stats(processed_docs)
            
            processing_time = time.time() - start_time
            
            logger.info(f"Document processing completed in {processing_time:.2f} seconds")
            
            return {
                'status': 'success',
                'successful_docs': len(successful_docs),
                'failed_docs': len(failed_docs),
                'total_chunks': len(all_chunks),
                'chunks_indexed': len(chunk_ids),
                'processing_time': processing_time,
                'processing_stats': processing_stats,
                'embedding_model_info': self.embedding_engine.get_model_info()
            }
            
        except Exception as e:
            logger.error(f"Error in document processing and indexing: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'processing_time': time.time() - start_time if 'start_time' in locals() else 0
            }
    
    def query(self, question: str, target_language: str = 'en', 
              top_k: int = None, include_sources: bool = True) -> Dict[str, Any]:
        """
        Process a query and return relevant answers with sources.
        
        Args:
            question (str): User's question in any supported language
            target_language (str): Language for the response
            top_k (int): Number of top results to retrieve
            include_sources: Whether to include source documents
            
        Returns:
            Dict[str, Any]: Query response with answer and sources
        """
        try:
            start_time = time.time()
            
            # Detect question language
            question_language = utils.detect_language(question)
            logger.info(f"Question language detected: {question_language}")
            
            # Generate embedding for the question
            question_embedding = self.embedding_engine.encode_text(question)
            
            # Search for relevant documents
            top_k = top_k or config.TOP_K_RESULTS
            search_results = self.vector_db.search(
                question_embedding, 
                top_k=top_k,
                filter_metadata=None  # No language filter for cross-lingual search
            )
            
            if not search_results:
                return {
                    'status': 'no_results',
                    'answer': f"No relevant information found for your question.",
                    'sources': [],
                    'query_language': question_language,
                    'response_language': target_language,
                    'processing_time': time.time() - start_time
                }
            
            # Prepare context from search results
            context_chunks = []
            source_documents = []
            
            for result in search_results:
                if result['similarity_score'] >= config.SIMILARITY_THRESHOLD:
                    context_chunks.append(result['text'])
                    if include_sources:
                        source_documents.append({
                            'id': result['id'],
                            'text': result['text'][:300] + '...' if len(result['text']) > 300 else result['text'],
                            'metadata': result['metadata'],
                            'similarity_score': result['similarity_score']
                        })
            
            if not context_chunks:
                return {
                    'status': 'low_relevance',
                    'answer': f"Found some information but relevance is below threshold.",
                    'sources': source_documents,
                    'query_language': question_language,
                    'response_language': target_language,
                    'processing_time': time.time() - start_time
                }
            
            # Generate answer based on context
            answer = self._generate_answer(question, context_chunks, question_language)
            
            # Translate answer if needed
            if target_language != question_language:
                translation_result = self.translation_service.translate_text(
                    answer, target_language, question_language
                )
                if translation_result['status'] == 'success':
                    answer = translation_result['translated_text']
                    logger.info(f"Answer translated from {question_language} to {target_language}")
                else:
                    logger.warning(f"Translation failed: {translation_result.get('error', 'Unknown error')}")
            
            # Translate source documents if needed
            if include_sources and target_language != question_language:
                for source in source_documents:
                    if source['metadata'].get('language') != target_language:
                        # Translate the source text preview
                        source_translation = self.translation_service.translate_text(
                            source['text'], target_language, source['metadata'].get('language', 'en')
                        )
                        if source_translation['status'] == 'success':
                            source['text'] = source_translation['translated_text']
                            source['translated'] = True
            
            processing_time = time.time() - start_time
            
            return {
                'status': 'success',
                'answer': answer,
                'sources': source_documents,
                'query_language': question_language,
                'response_language': target_language,
                'processing_time': processing_time,
                'relevance_threshold': config.SIMILARITY_THRESHOLD,
                'total_results': len(search_results),
                'relevant_results': len(context_chunks)
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                'status': 'error',
                'message': f"Error processing your question: {str(e)}",
                'processing_time': time.time() - start_time if 'start_time' in locals() else 0
            }
    
    def _generate_answer(self, question: str, context_chunks: List[str], 
                        question_language: str) -> str:
        """
        Generate an answer based on the question and context chunks.
        
        Args:
            question (str): User's question
            context_chunks (List[str]): Relevant context chunks
            question_language (str): Language of the question
            
        Returns:
            str: Generated answer
        """
        try:
            # Combine context chunks
            combined_context = "\n\n".join(context_chunks)
            
            # Simple answer generation based on context
            # In a production system, you might use a more sophisticated LLM here
            
            # Check if the question is asking for specific information
            question_lower = question.lower()
            
            if any(word in question_lower for word in ['what', 'how', 'why', 'when', 'where']):
                # Extract relevant information from context
                answer = self._extract_relevant_info(question, combined_context)
            elif any(word in question_lower for word in ['symptoms', 'signs', 'indicators']):
                # Look for symptom-related information
                answer = self._extract_symptom_info(combined_context)
            elif any(word in question_lower for word in ['treatment', 'therapy', 'medication']):
                # Look for treatment-related information
                answer = self._extract_treatment_info(combined_context)
            else:
                # General information extraction
                answer = self._extract_general_info(combined_context)
            
            # Ensure answer is not too long
            if len(answer) > 1000:
                answer = answer[:1000] + "..."
            
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "I'm sorry, I couldn't generate a proper answer based on the available information."
    
    def _extract_relevant_info(self, question: str, context: str) -> str:
        """Extract relevant information based on the question."""
        # Simple keyword-based extraction
        question_words = set(question.lower().split())
        context_sentences = context.split('. ')
        
        relevant_sentences = []
        for sentence in context_sentences:
            sentence_words = set(sentence.lower().split())
            if question_words.intersection(sentence_words):
                relevant_sentences.append(sentence)
        
        if relevant_sentences:
            return '. '.join(relevant_sentences[:3]) + '.'
        else:
            return context[:500] + "..."
    
    def _extract_symptom_info(self, context: str) -> str:
        """Extract symptom-related information from context."""
        symptom_keywords = ['symptom', 'sign', 'indicator', 'warning', 'alert']
        context_lower = context.lower()
        
        # Find sentences containing symptom information
        sentences = context.split('. ')
        symptom_sentences = []
        
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in symptom_keywords):
                symptom_sentences.append(sentence)
        
        if symptom_sentences:
            return ' '.join(symptom_sentences[:3]) + '.'
        else:
            return "Based on the available information, I couldn't find specific symptom details."
    
    def _extract_treatment_info(self, context: str) -> str:
        """Extract treatment-related information from context."""
        treatment_keywords = ['treatment', 'therapy', 'medication', 'care', 'management', 'strategy']
        context_lower = context.lower()
        
        # Find sentences containing treatment information
        sentences = context.split('. ')
        treatment_sentences = []
        
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in treatment_keywords):
                treatment_sentences.append(sentence)
        
        if treatment_sentences:
            return ' '.join(treatment_sentences[:3]) + '.'
        else:
            return "Based on the available information, I couldn't find specific treatment details."
    
    def _extract_general_info(self, context: str) -> str:
        """Extract general information from context."""
        # Take the first few sentences that seem most informative
        sentences = context.split('. ')
        informative_sentences = []
        
        for sentence in sentences[:3]:  # Take first 3 sentences
            if len(sentence.strip()) > 20:  # Only sentences with substantial content
                informative_sentences.append(sentence)
        
        if informative_sentences:
            return '. '.join(informative_sentences) + '.'
        else:
            return context[:500] + "..."
    
    def batch_query(self, questions: List[str], target_language: str = 'en') -> List[Dict[str, Any]]:
        """
        Process multiple queries in batch.
        
        Args:
            questions (List[str]): List of questions to process
            target_language (str): Language for responses
            
        Returns:
            List[Dict[str, Any]]: List of query responses
        """
        try:
            results = []
            for i, question in enumerate(questions):
                logger.info(f"Processing question {i+1}/{len(questions)}")
                result = self.query(question, target_language)
                result['question_index'] = i
                result['question'] = question
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch query processing: {e}")
            return []
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status information.
        
        Returns:
            Dict[str, Any]: System status information
        """
        try:
            return {
                'rag_engine_status': 'active',
                'document_processor': {
                    'status': 'active',
                    'supported_formats': list(self.document_processor.supported_extensions.keys())
                },
                'embedding_engine': self.embedding_engine.get_model_info(),
                'translation_service': self.translation_service.get_service_status(),
                'vector_database': self.vector_db.get_database_info(),
                'system_config': {
                    'supported_languages': len(config.SUPPORTED_LANGUAGES),
                    'chunk_size': config.CHUNK_SIZE,
                    'chunk_overlap': config.CHUNK_OVERLAP,
                    'top_k_results': config.TOP_K_RESULTS,
                    'similarity_threshold': config.SIMILARITY_THRESHOLD
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def add_sample_documents(self) -> Dict[str, Any]:
        """
        Add sample healthcare documents for testing purposes.
        
        Returns:
            Dict[str, Any]: Results of adding sample documents
        """
        try:
            logger.info("Adding sample healthcare documents...")
            
            # Create sample documents
            sample_docs = self.document_processor.create_sample_documents()
            
            if not sample_docs:
                return {
                    'status': 'error',
                    'message': 'Failed to create sample documents'
                }
            
            # Process and index sample documents
            result = self.process_and_index_documents([])  # Empty list since we're using sample docs
            
            # Override with sample document processing
            all_chunks = []
            all_metadata = []
            
            for doc in sample_docs:
                chunks = doc.get('chunks', [])
                for i, chunk in enumerate(chunks):
                    all_chunks.append(chunk)
                    chunk_metadata = doc['metadata'].copy()
                    chunk_metadata.update({
                        'chunk_index': i,
                        'chunk_text': chunk[:200] + '...' if len(chunk) > 200 else chunk
                    })
                    all_metadata.append(chunk_metadata)
            
            # Generate embeddings
            embeddings, enhanced_metadata = self.embedding_engine.encode_chunks(
                all_chunks, all_metadata
            )
            
            # Add to vector database
            chunk_ids = self.vector_db.add_chunks(all_chunks, embeddings, enhanced_metadata)
            
            return {
                'status': 'success',
                'sample_documents_added': len(sample_docs),
                'total_chunks': len(all_chunks),
                'chunks_indexed': len(chunk_ids),
                'languages': list(set(doc['language'] for doc in sample_docs))
            }
            
        except Exception as e:
            logger.error(f"Error adding sample documents: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def evaluate_system(self, test_queries: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Evaluate system performance with test queries.
        
        Args:
            test_queries (List[Dict[str, str]]): List of test queries with expected answers
            
        Returns:
            Dict[str, Any]: Evaluation results
        """
        try:
            evaluation_results = {
                'total_queries': len(test_queries),
                'successful_queries': 0,
                'failed_queries': 0,
                'average_response_time': 0.0,
                'language_coverage': set(),
                'query_results': []
            }
            
            total_response_time = 0.0
            
            for i, test_case in enumerate(test_queries):
                query = test_case['query']
                expected_language = test_case.get('language', 'en')
                
                start_time = time.time()
                result = self.query(query, expected_language)
                response_time = time.time() - start_time
                
                total_response_time += response_time
                
                # Record language coverage
                if result.get('query_language'):
                    evaluation_results['language_coverage'].add(result['query_language'])
                
                # Evaluate success
                if result['status'] == 'success':
                    evaluation_results['successful_queries'] += 1
                else:
                    evaluation_results['failed_queries'] += 1
                
                # Store query result
                evaluation_results['query_results'].append({
                    'query': query,
                    'result': result,
                    'response_time': response_time
                })
            
            # Calculate averages
            if evaluation_results['total_queries'] > 0:
                evaluation_results['average_response_time'] = total_response_time / evaluation_results['total_queries']
                evaluation_results['success_rate'] = evaluation_results['successful_queries'] / evaluation_results['total_queries']
            
            evaluation_results['language_coverage'] = list(evaluation_results['language_coverage'])
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Error in system evaluation: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
