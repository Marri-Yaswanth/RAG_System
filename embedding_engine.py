"""
Embedding Engine for Multi-Language RAG System
Handles text embeddings using multi-language sentence transformers
"""
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
import torch
import config

logger = logging.getLogger(__name__)

class EmbeddingEngine:
    """
    Handles text embeddings for multiple languages using sentence transformers.
    Provides cross-lingual embedding capabilities and similarity search.
    """
    
    def __init__(self, model_name: str = None):
        """
        Initialize the embedding engine.
        
        Args:
            model_name (str): Name of the sentence transformer model to use
        """
        self.model_name = model_name or config.EMBEDDING_MODEL_NAME
        self.model = None
        self.device = self._get_device()
        self._load_model()
    
    def _get_device(self) -> str:
        """Get the best available device for embeddings."""
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    
    def _load_model(self):
        """Load the sentence transformer model."""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            logger.info(f"Using device: {self.device}")
            
            self.model = SentenceTransformer(self.model_name, device=self.device)
            
            # Test the model with a simple embedding
            test_text = "Hello world"
            test_embedding = self.model.encode(test_text)
            logger.info(f"Model loaded successfully. Test embedding shape: {test_embedding.shape}")
            
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            raise
    
    def encode_text(self, text: str, normalize: bool = True) -> np.ndarray:
        """
        Encode a single text string to embedding vector.
        
        Args:
            text (str): Input text to encode
            normalize (bool): Whether to normalize the embedding vector
            
        Returns:
            np.ndarray: Embedding vector
        """
        try:
            if not text or not text.strip():
                raise ValueError("Text cannot be empty")
            
            # Encode the text
            embedding = self.model.encode(text, normalize_embeddings=normalize)
            return embedding
            
        except Exception as e:
            logger.error(f"Error encoding text: {e}")
            raise
    
    def encode_texts(self, texts: List[str], normalize: bool = True, 
                     batch_size: int = 32) -> np.ndarray:
        """
        Encode multiple text strings to embedding vectors.
        
        Args:
            texts (List[str]): List of input texts to encode
            normalize (bool): Whether to normalize the embedding vectors
            batch_size (int): Batch size for processing
            
        Returns:
            np.ndarray: Matrix of embedding vectors
        """
        try:
            if not texts:
                raise ValueError("Texts list cannot be empty")
            
            # Filter out empty texts
            valid_texts = [text for text in texts if text and text.strip()]
            if not valid_texts:
                raise ValueError("No valid texts found")
            
            # Encode texts in batches
            embeddings = self.model.encode(valid_texts, 
                                         normalize_embeddings=normalize,
                                         batch_size=batch_size,
                                         show_progress_bar=len(valid_texts) > 100)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error encoding texts: {e}")
            raise
    
    def encode_chunks(self, chunks: List[str], metadata: List[Dict] = None, 
                     normalize: bool = True) -> Tuple[np.ndarray, List[Dict]]:
        """
        Encode text chunks and return embeddings with metadata.
        
        Args:
            chunks (List[str]): List of text chunks to encode
            metadata (List[Dict]): Optional metadata for each chunk
            normalize (bool): Whether to normalize the embedding vectors
            
        Returns:
            Tuple[np.ndarray, List[Dict]]: Embeddings and enhanced metadata
        """
        try:
            if not chunks:
                return np.array([]), []
            
            # Encode chunks
            embeddings = self.encode_texts(chunks, normalize=normalize)
            
            # Prepare metadata
            if metadata is None:
                metadata = [{} for _ in chunks]
            
            # Ensure metadata list has same length as chunks
            if len(metadata) != len(chunks):
                metadata = metadata[:len(chunks)] + [{} for _ in range(len(chunks) - len(metadata))]
            
            # Add embedding information to metadata
            enhanced_metadata = []
            for i, (chunk, meta) in enumerate(zip(chunks, metadata)):
                enhanced_meta = meta.copy()
                enhanced_meta.update({
                    'chunk_index': i,
                    'chunk_length': len(chunk),
                    'embedding_dimension': embeddings.shape[1] if len(embeddings) > 0 else 0,
                    'chunk_text': chunk[:200] + '...' if len(chunk) > 200 else chunk  # Truncate for storage
                })
                enhanced_metadata.append(enhanced_meta)
            
            return embeddings, enhanced_metadata
            
        except Exception as e:
            logger.error(f"Error encoding chunks: {e}")
            raise
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embedding vectors.
        
        Args:
            embedding1 (np.ndarray): First embedding vector
            embedding2 (np.ndarray): Second embedding vector
            
        Returns:
            float: Cosine similarity score between -1 and 1
        """
        try:
            if embedding1.shape != embedding2.shape:
                raise ValueError("Embedding vectors must have the same shape")
            
            # Calculate cosine similarity
            similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def find_most_similar(self, query_embedding: np.ndarray, 
                          candidate_embeddings: np.ndarray, 
                          top_k: int = 5) -> Tuple[List[int], List[float]]:
        """
        Find the most similar embeddings to a query embedding.
        
        Args:
            query_embedding (np.ndarray): Query embedding vector
            candidate_embeddings (np.ndarray): Matrix of candidate embedding vectors
            top_k (int): Number of top similar embeddings to return
            
        Returns:
            Tuple[List[int], List[float]]: Indices and similarity scores of top matches
        """
        try:
            if len(candidate_embeddings) == 0:
                return [], []
            
            # Calculate similarities
            similarities = []
            for i, candidate in enumerate(candidate_embeddings):
                similarity = self.calculate_similarity(query_embedding, candidate)
                similarities.append((i, similarity))
            
            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Get top-k results
            top_indices = [idx for idx, _ in similarities[:top_k]]
            top_scores = [score for _, score in similarities[:top_k]]
            
            return top_indices, top_scores
            
        except Exception as e:
            logger.error(f"Error finding most similar embeddings: {e}")
            return [], []
    
    def semantic_search(self, query: str, candidate_texts: List[str], 
                       top_k: int = 5, threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Perform semantic search using text similarity.
        
        Args:
            query (str): Search query text
            candidate_texts (List[str]): List of candidate texts to search through
            top_k (int): Number of top results to return
            threshold (float): Minimum similarity threshold
            
        Returns:
            List[Dict[str, Any]]: List of search results with scores and metadata
        """
        try:
            if not query or not candidate_texts:
                return []
            
            # Encode query
            query_embedding = self.encode_text(query)
            
            # Encode candidates
            candidate_embeddings = self.encode_texts(candidate_texts)
            
            # Find most similar
            top_indices, top_scores = self.find_most_similar(
                query_embedding, candidate_embeddings, top_k=len(candidate_texts)
            )
            
            # Filter by threshold and format results
            results = []
            for idx, score in zip(top_indices, top_scores):
                if score >= threshold:
                    results.append({
                        'index': idx,
                        'text': candidate_texts[idx],
                        'similarity_score': score,
                        'rank': len(results) + 1
                    })
            
            # Return top-k results
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []
    
    def batch_semantic_search(self, queries: List[str], candidate_texts: List[str],
                             top_k: int = 5, threshold: float = 0.0) -> List[List[Dict[str, Any]]]:
        """
        Perform batch semantic search for multiple queries.
        
        Args:
            queries (List[str]): List of search query texts
            candidate_texts (List[str]): List of candidate texts to search through
            top_k (int): Number of top results to return per query
            threshold (float): Minimum similarity threshold
            
        Returns:
            List[List[Dict[str, Any]]]: List of search results for each query
        """
        try:
            if not queries or not candidate_texts:
                return []
            
            # Encode all queries
            query_embeddings = self.encode_texts(queries)
            
            # Encode candidates
            candidate_embeddings = self.encode_texts(candidate_texts)
            
            # Process each query
            all_results = []
            for i, query_embedding in enumerate(query_embeddings):
                query_results = []
                top_indices, top_scores = self.find_most_similar(
                    query_embedding, candidate_embeddings, top_k=len(candidate_texts)
                )
                
                # Filter by threshold and format results
                for idx, score in zip(top_indices, top_scores):
                    if score >= threshold:
                        query_results.append({
                            'query_index': i,
                            'query': queries[i],
                            'index': idx,
                            'text': candidate_texts[idx],
                            'similarity_score': score,
                            'rank': len(query_results) + 1
                        })
                
                all_results.append(query_results[:top_k])
            
            return all_results
            
        except Exception as e:
            logger.error(f"Error in batch semantic search: {e}")
            return []
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding vectors."""
        if self.model is None:
            return 0
        return self.model.get_sentence_embedding_dimension()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if self.model is None:
            return {}
        
        return {
            'model_name': self.model_name,
            'embedding_dimension': self.get_embedding_dimension(),
            'device': self.device,
            'max_seq_length': getattr(self.model, 'max_seq_length', 'Unknown'),
            'model_type': type(self.model).__name__
        }
    
    def validate_embeddings(self, embeddings: np.ndarray) -> bool:
        """
        Validate that embeddings are properly formatted.
        
        Args:
            embeddings (np.ndarray): Embeddings to validate
            
        Returns:
            bool: True if embeddings are valid, False otherwise
        """
        try:
            if not isinstance(embeddings, np.ndarray):
                return False
            
            if len(embeddings.shape) != 2:
                return False
            
            expected_dim = self.get_embedding_dimension()
            if embeddings.shape[1] != expected_dim:
                return False
            
            # Check for NaN or infinite values
            if np.any(np.isnan(embeddings)) or np.any(np.isinf(embeddings)):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating embeddings: {e}")
            return False
