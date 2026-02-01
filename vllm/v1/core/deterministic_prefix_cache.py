"""
Deterministic Prefix Cache: Ensures R1 (cache miss) produces identical
results to R2+ (cache hits) by splitting the first request into two passes.

Key Idea:
- R1: Compute in TWO passes (prefix alone, then suffix with cached prefix)
- R2+: Natural cache hit (prefix cached, compute suffix)
- Result: R1 == R2 == R3... (fully deterministic)
"""

import os
from typing import Optional
from vllm.logger import init_logger

logger = init_logger(__name__)


class DeterministicPrefixCache:
    """
    Manages deterministic prefix caching by ensuring first requests
    compute in the same pattern as subsequent cache-hit requests.
    """
    
    def __init__(self, enable: bool = False, block_size: int = 16):
        self.enable = enable or os.getenv("VLLM_DETERMINISTIC_PREFIX_CACHE", "0") == "1"
        self.block_size = block_size
        self.first_request_tracker = {}  # request_id -> is_first_for_prefix
        
        if self.enable:
            logger.info(
                "Deterministic Prefix Cache ENABLED with block_size=%d. "
                "First requests will use two-pass computation for determinism.",
                block_size
            )
    
    def should_split_request(self, request_id: str, num_tokens: int, 
                            cached_tokens: int) -> tuple[bool, int]:
        """
        Determine if a request should be split for deterministic caching.
        
        Args:
            request_id: Unique request identifier
            num_tokens: Total number of tokens in request
            cached_tokens: Number of tokens already cached
            
        Returns:
            (should_split, split_point): Whether to split and where
        """
        if not self.enable:
            return False, 0
        
        # Only split if this is a prefill request (multiple tokens)
        if num_tokens <= 1:
            return False, 0
        
        # If we have a cache hit, no need to split (already deterministic)
        if cached_tokens > 0:
            return False, 0
        
        # This is a cache MISS with multiple tokens
        # Split at block boundary to match how cache hits will work
        split_point = (num_tokens // self.block_size) * self.block_size
        
        # Only split if we have a meaningful prefix
        if split_point >= self.block_size and split_point < num_tokens:
            logger.debug(
                "[DETERMINISTIC] Splitting request %s: tokens=%d → "
                "prefix=[0:%d], suffix=[%d:%d]",
                request_id[-12:], num_tokens, split_point, split_point, num_tokens
            )
            return True, split_point
        
        return False, 0
    
    def mark_as_two_pass(self, request_id: str):
        """Mark that this request is using two-pass computation."""
        self.first_request_tracker[request_id] = True
    
    def is_two_pass(self, request_id: str) -> bool:
        """Check if this request is using two-pass computation."""
        return self.first_request_tracker.get(request_id, False)
    
    def cleanup(self, request_id: str):
        """Clean up tracking data for completed request."""
        self.first_request_tracker.pop(request_id, None)


# Global instance
_deterministic_cache: Optional[DeterministicPrefixCache] = None


def get_deterministic_cache() -> DeterministicPrefixCache:
    """Get or create the global deterministic cache instance."""
    global _deterministic_cache
    if _deterministic_cache is None:
        # Check if enabled via environment variable
        enable = os.getenv("VLLM_DETERMINISTIC_PREFIX_CACHE", "0") == "1"
        block_size = int(os.getenv("VLLM_DETERMINISTIC_CACHE_BLOCK_SIZE", "16"))
        _deterministic_cache = DeterministicPrefixCache(enable=enable, block_size=block_size)
    return _deterministic_cache


def should_use_deterministic_mode() -> bool:
    """Check if deterministic prefix caching is enabled."""
    return os.getenv("VLLM_DETERMINISTIC_PREFIX_CACHE", "0") == "1"
