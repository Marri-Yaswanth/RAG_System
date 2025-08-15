"""
Vector Database using FAISS for Multi-Language RAG System
Provides FAISS-based vector storage and similarity search
"""
import logging
import os
import pickle
import time
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import faiss
import config
import utils

logger = logging.getLogger(__name__)

class VectorDatabase:
    """
    FAISS-based vector database for storing and retrieving document embeddings.
    Provides efficient similarity search and metadata management.
    """
    
    def __init__(self, persist_directory: str = None):
        """
        Initialize the FAISS vector database.
        
        Args:
            persist_directory (str): Directory to persist the database
        """
        self.persist_directory = persist_directory or config.CHROMA_PERSIST_DIRECTORY
        self.index = None
        self.metadata = []
        self.document_ids = []
        
        # Create persist directory if it doesn't exist
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # FAISS index file paths
        self.index_file = os.path.join(self.persist_directory, 'faiss_index.bin')
        self.metadata_file = os.path.join(self.persist_directory, 'metadata.pkl')
        self.ids_file = os.path.join(self.persist_directory, 'document_ids.pkl')
        
        # Load existing index if available
        self._load_index()
        
        logger.info(f"FAISS Vector Database initialized at {self.persist_directory}")
    
    def _load_index(self):
        """Load existing FAISS index and metadata if available."""
        try:
            if os.path.exists(self.index_file) and os.path.exists(self.metadata_file):
                # Load FAISS index
                self.index = faiss.read_index(self.index_file)
                
                # Load metadata
                with open(self.metadata_file, 'rb') as f:
                    self.metadata = pickle.load(f)
                
                # Load document IDs
                with open(self.ids_file, 'rb') as f:
                    self.document_ids = pickle.load(f)
                
                logger.info(f"Loaded existing index with {len(self.metadata)} documents")
            else:
                # Create new index
                self._create_new_index()
                
        except Exception as e:
            logger.warning(f"Failed to load existing index: {e}")
            self._create_new_index()
    
    def _create_new_index(self):
        """Create a new FAISS index."""
        # Create a flat index for L2 distance
        dimension = 384  # Default dimension for sentence-transformers
        self.index = faiss.IndexFlatL2(dimension)
        self.metadata = []
        self.document_ids = []
        logger.info("Created new FAISS index")
    
    def _save_index(self):
        """Save the current index and metadata to disk."""
        try:
            if self.index is not None:
                # Save FAISS index
                faiss.write_index(self.index, self.index_file)
                
                # Save metadata
                with open(self.metadata_file, 'wb') as f:
                    pickle.dump(self.metadata, f)
                
                # Save document IDs
                with open(self.ids_file, 'wb') as f:
                    pickle.dump(self.document_ids, f)
                
                logger.info("Index saved successfully")
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
    
    def add_documents(self, documents: List[Dict[str, Any]], 
                     embeddings: np.ndarray) -> List[str]:
        """
        Add documents with their embeddings to the database.
        
        Args:
            documents (List[Dict]): List of document dictionaries
            embeddings (np.ndarray): Document embeddings
            
        Returns:
            List[str]: List of document IDs
        """
        try:
            if self.index is None:
                self._create_new_index()
            
            # Add embeddings to FAISS index
            self.index.add(embeddings.astype(np.float32))
            
            # Store metadata
            for doc in documents:
                self.metadata.append(doc)
                self.document_ids.append(doc.get('id', f"doc_{len(self.document_ids)}"))
            
            # Save to disk
            self._save_index()
            
            logger.info(f"Added {len(documents)} documents to database")
            return self.document_ids[-len(documents):]
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return []
    
    def add_chunks(self, chunks: List[str], embeddings: np.ndarray, 
                   metadata: List[Dict[str, Any]]) -> List[str]:
        """
        Add document chunks with their embeddings to the database.
        
        Args:
            chunks (List[str]): List of text chunks
            embeddings (np.ndarray): Chunk embeddings
            metadata (List[Dict]): List of metadata for each chunk
            
        Returns:
            List[str]: List of chunk IDs
        """
        try:
            if self.index is None:
                self._create_new_index()
            
            # Add embeddings to FAISS index
            self.index.add(embeddings.astype(np.float32))
            
            # Store metadata with chunk text
            chunk_ids = []
            for i, (chunk, meta) in enumerate(zip(chunks, metadata)):
                chunk_id = f"chunk_{len(self.document_ids)}_{i}"
                chunk_metadata = meta.copy()
                chunk_metadata.update({
                    'chunk_text': chunk,
                    'chunk_id': chunk_id
                })
                
                self.metadata.append(chunk_metadata)
                self.document_ids.append(chunk_id)
                chunk_ids.append(chunk_id)
            
            # Save to disk
            self._save_index()
            
            logger.info(f"Added {len(chunks)} chunks to database")
            return chunk_ids
            
        except Exception as e:
            logger.error(f"Error adding chunks: {e}")
            return []
    
    def search(self, query_embedding: np.ndarray, top_k: int = None, 
               filter_metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents/chunks.
        
        Args:
            query_embedding (np.ndarray): Query embedding
            top_k (int): Number of top results to return
            filter_metadata (Dict): Metadata filters (not implemented in FAISS)
            
        Returns:
            List[Dict]: Search results with similarity scores
        """
        try:
            if self.index is None or self.index.ntotal == 0:
                return []
            
            top_k = top_k or config.TOP_K_RESULTS
            top_k = min(top_k, self.index.ntotal)
            
            # Search FAISS index
            query_embedding = query_embedding.astype(np.float32).reshape(1, -1)
            distances, indices = self.index.search(query_embedding, top_k)
            
            # Convert distances to similarity scores (1 / (1 + distance))
            similarity_scores = 1 / (1 + distances[0])
            
            # Prepare results
            results = []
            for i, (idx, score) in enumerate(zip(indices[0], similarity_scores)):
                if idx < len(self.metadata):
                    result = {
                        'id': self.document_ids[idx],
                        'text': self.metadata[idx].get('chunk_text', ''),
                        'metadata': self.metadata[idx],
                        'similarity_score': float(score),
                        'rank': i + 1
                    }
                    results.append(result)
            
            # Sort by similarity score (descending)
            results.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            logger.info(f"Search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error during search: {e}")
            return []
    
    def get_document_by_id(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a document by its ID.
        
        Args:
            document_id (str): Document ID
            
        Returns:
            Optional[Dict]: Document data or None if not found
        """
        try:
            if document_id in self.document_ids:
                idx = self.document_ids.index(document_id)
                return self.metadata[idx]
            return None
        except Exception as e:
            logger.error(f"Error retrieving document: {e}")
            return None
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Dict[str, Any]: Database statistics
        """
        try:
            stats = {
                'total_documents': len(self.metadata),
                'total_chunks': len(self.metadata),
                'index_size': self.index.ntotal if self.index else 0,
                'persist_directory': self.persist_directory,
                'index_file_exists': os.path.exists(self.index_file),
                'metadata_file_exists': os.path.exists(self.metadata_file)
            }
            
            # Language distribution
            languages = {}
            for meta in self.metadata:
                lang = meta.get('language', 'unknown')
                languages[lang] = languages.get(lang, 0) + 1
            
            stats['language_distribution'] = languages
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting database info: {e}")
            return {}
    
    def get_database_info(self) -> Dict[str, Any]:
        """
        Get comprehensive database information including collection stats and system config.
        
        Returns:
            Dict[str, Any]: Complete database information
        """
        try:
            collection_stats = self.get_collection_stats()
            
            database_info = {
                'collection_stats': collection_stats,
                'system_config': {
                    'persist_directory': self.persist_directory,
                    'index_type': type(self.index).__name__ if self.index else 'None',
                    'embedding_dimension': self.index.d if self.index else 384,
                    'total_vectors': self.index.ntotal if self.index else 0,
                    'is_trained': self.index.is_trained if self.index else False
                },
                'status': 'active' if self.index else 'inactive'
            }
            
            return database_info
            
        except Exception as e:
            logger.error(f"Error getting database info: {e}")
            return {
                'collection_stats': {},
                'system_config': {},
                'status': 'error'
            }
    
    def clear_database(self):
        """Clear all data from the database."""
        try:
            self.index = None
            self.metadata = []
            self.document_ids = []
            
            # Remove files
            for file_path in [self.index_file, self.metadata_file, self.ids_file]:
                if os.path.exists(file_path):
                    os.remove(file_path)
            
            # Create new index
            self._create_new_index()
            
            logger.info("Database cleared successfully")
            
        except Exception as e:
            logger.error(f"Error clearing database: {e}")
    
    def __len__(self):
        """Return the number of documents in the database."""
        return len(self.metadata)
