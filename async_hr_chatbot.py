"""
Async version of HR Chatbot functionality.

This module provides asynchronous implementations of the core
HR Chatbot functions for improved performance and concurrency.
"""

import asyncio
import logging
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class AsyncHRChatbot:
    """
    Asynchronous HR Chatbot for improved performance and concurrency.
    """
    
    def __init__(self, max_workers: int = 4):
        """
        Initialize the async chatbot.
        
        Args:
            max_workers: Maximum number of worker threads for CPU-bound tasks
        """
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        self.executor.shutdown(wait=True)
    
    async def process_query_async(self, query: str) -> str:
        """
        Asynchronously process a query.
        
        Args:
            query: User's question
            
        Returns:
            Processed response
        """
        def process_query():
            try:
                # Simulate processing
                logger.info(f"Processing query: {query[:50]}...")
                return f"Processed: {query}"
            except Exception as e:
                logger.error(f"Error processing query: {str(e)}")
                raise
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, process_query)


# Convenience functions
async def process_query_async(query: str) -> str:
    """
    Async convenience function to process a query.
    
    Args:
        query: User's question
        
    Returns:
        Processed response
    """
    async with AsyncHRChatbot() as chatbot:
        return await chatbot.process_query_async(query)


# Example usage
async def main():
    """
    Example usage of async HR chatbot.
    """
    async with AsyncHRChatbot() as chatbot:
        # Process a query
        result = await chatbot.process_query_async("What is the vacation policy?")
        print(f"Result: {result}")


if __name__ == "__main__":
    asyncio.run(main())