"""Maintenance utilities for RAG indexes"""
import logging
import time
from typing import Optional, List, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

class MaintenanceUtils:
    """Utilities for maintaining and optimizing RAG indexes"""
    
    @staticmethod
    def rebuild_bm25_index(store, collection_name: str) -> bool:
        """
        Rebuild the BM25 index for a collection.
        
        Args:
            store: EnhancedStore instance
            collection_name: Name of collection to rebuild
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Rebuilding BM25 index for collection '{collection_name}'")
            
            # Get the collection
            collection = store.get_collection(collection_name)
            if not collection:
                logger.error(f"Collection '{collection_name}' not found")
                return False
                
            # Clear existing BM25 index
            if collection_name in store.bm25_index.collection_indexes:
                del store.bm25_index.collection_indexes[collection_name]
                
            # Process all documents and rebuild index
            all_chunks = []
            for document in collection.documents:
                chunks = store.document_processor.process_document(document)
                all_chunks.extend(chunks)
                
            # Add all chunks to BM25 index
            store.bm25_index.add_collection(collection_name, all_chunks)
            
            # Save updated index
            if not store.in_memory:
                store._save_bm25_index(collection_name)
                
            logger.info(f"Successfully rebuilt BM25 index for '{collection_name}' with {len(all_chunks)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Error rebuilding BM25 index for '{collection_name}': {e}")
            import traceback
            traceback.print_exc()
            return False
    
    @staticmethod
    def optimize_cache(store, max_age_hours: int = 24) -> int:
        """
        Remove old entries from result cache.
        
        Args:
            store: EnhancedStore instance
            max_age_hours: Maximum age of cache entries in hours
            
        Returns:
            Number of entries removed
        """
        try:
            current_time = time.time()
            max_age_seconds = max_age_hours * 3600
            
            # Identify old entries
            old_keys = []
            for key, entry in store.result_cache.items():
                if current_time - entry['timestamp'] > max_age_seconds:
                    old_keys.append(key)
                    
            # Remove old entries
            for key in old_keys:
                del store.result_cache[key]
                
            logger.info(f"Removed {len(old_keys)} old entries from result cache")
            return len(old_keys)
            
        except Exception as e:
            logger.error(f"Error optimizing cache: {e}")
            return 0
    
    @staticmethod
    def verify_collection_integrity(store, collection_name: str) -> Dict[str, Any]:
        """
        Verify the integrity of a collection's indexes.
        
        Args:
            store: EnhancedStore instance
            collection_name: Name of collection to verify
            
        Returns:
            Report of verification results
        """
        report = {
            'collection': collection_name,
            'timestamp': time.time(),
            'vector_index': {'status': 'unknown', 'count': 0},
            'bm25_index': {'status': 'unknown', 'count': 0},
            'documents': {'status': 'unknown', 'count': 0},
            'issues': []
        }
        
        try:
            # Check if collection exists
            collection = store.get_collection(collection_name)
            if not collection:
                report['issues'].append(f"Collection '{collection_name}' not found")
                return report
                
            # Check document count
            doc_count = len(collection.documents)
            report['documents']['count'] = doc_count
            report['documents']['status'] = 'ok'
            
            # Check vector index
            try:
                vector_count = store._connection.count(
                    collection_name=collection_name
                ).count
                report['vector_index']['count'] = vector_count
                report['vector_index']['status'] = 'ok'
            except Exception as ve:
                report['vector_index']['status'] = 'error'
                report['issues'].append(f"Vector index error: {str(ve)}")
                
            # Check BM25 index
            if collection_name in store.bm25_index.collection_indexes:
                bm25_count = len(store.bm25_index.collection_indexes[collection_name].chunks)
                report['bm25_index']['count'] = bm25_count
                report['bm25_index']['status'] = 'ok'
                
                # Check for discrepancies
                if bm25_count != vector_count and report['vector_index']['status'] == 'ok':
                    report['issues'].append(
                        f"Index count mismatch: BM25={bm25_count}, Vector={vector_count}"
                    )
            else:
                report['bm25_index']['status'] = 'missing'
                report['issues'].append(f"BM25 index not found for collection '{collection_name}'")
                
            # Overall status
            if not report['issues']:
                report['status'] = 'ok'
            elif report['vector_index']['status'] == 'ok' and report['documents']['status'] == 'ok':
                report['status'] = 'partial'
            else:
                report['status'] = 'error'
                
            return report
            
        except Exception as e:
            logger.error(f"Error verifying collection '{collection_name}': {e}")
            report['status'] = 'error'
            report['issues'].append(f"Verification error: {str(e)}")
            return report