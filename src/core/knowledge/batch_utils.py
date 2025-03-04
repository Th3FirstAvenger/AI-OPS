"""Utilities for batch processing in RAG operations"""
import logging
from typing import List, Any, Callable, TypeVar, Iterator

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')

def batch_process(items: List[T], 
                process_fn: Callable[[List[T]], List[R]], 
                batch_size: int = 10) -> List[R]:
    """
    Process a list of items in batches.
    
    Args:
        items: List of items to process
        process_fn: Function that processes a batch of items
        batch_size: Size of each batch
        
    Returns:
        List of processed results
    """
    results = []
    
    # Create batches
    batches = list(create_batches(items, batch_size))
    total_batches = len(batches)
    
    logger.info(f"Processing {len(items)} items in {total_batches} batches of size {batch_size}")
    
    # Process each batch
    for i, batch in enumerate(batches):
        logger.debug(f"Processing batch {i+1}/{total_batches}")
        batch_results = process_fn(batch)
        results.extend(batch_results)
        
    return results

def create_batches(items: List[T], batch_size: int) -> Iterator[List[T]]:
    """
    Split a list of items into batches.
    
    Args:
        items: List of items to split
        batch_size: Size of each batch
        
    Returns:
        Iterator of batches
    """
    for i in range(0, len(items), batch_size):
        yield items[i:i+batch_size]