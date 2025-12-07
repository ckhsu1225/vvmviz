"""
Cache Management Module

Provides a unified caching system for VVMViz with:
- Two-layer caching (dataset-level and frame-level)
- LRU eviction policy
- Thread-safe operations
- Background prefetching with ThreadPoolExecutor
- Cache metrics and monitoring
"""

import threading
import logging
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, Future
from time import time

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class FrameRequest:
    """
    Specification for loading a single frame of data.

    This encapsulates all parameters needed to load and cache a frame,
    including the main variable and overlay configurations.
    """
    sim_path: str
    var_name: str
    t_range: Tuple[int, int]
    z_range: Tuple[Any, ...]  # Can be (min, max) or ('index', idx, idx)
    x_range: Tuple[int, int]
    y_range: Tuple[int, int]

    # Overlay configurations
    wind_enabled: bool = False
    use_surface: bool = False
    contour_enabled: bool = False
    contour_var: Optional[str] = None

    def cache_key(self) -> Tuple:
        """
        Generate cache key for this request.

        The cache key identifies a unique frame based on variable and ranges.
        Overlay settings are not included in the key as they are part of the
        cached bundle.

        Returns
        -------
        tuple
            Cache key: (var_name, t_range, z_range)
        """
        return (self.var_name, self.t_range, self.z_range)


@dataclass
class CacheMetrics:
    """
    Track cache performance metrics.

    Attributes
    ----------
    hits : int
        Number of cache hits
    misses : int
        Number of cache misses
    prefetch_success : int
        Number of successful prefetch operations
    prefetch_failure : int
        Number of failed prefetch operations
    prefetch_cancelled : int
        Number of cancelled prefetch operations
    total_load_time : float
        Cumulative time spent loading data (seconds)
    """
    hits: int = 0
    misses: int = 0
    prefetch_success: int = 0
    prefetch_failure: int = 0
    prefetch_cancelled: int = 0
    total_load_time: float = 0.0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def average_load_time(self) -> float:
        """Calculate average load time."""
        total_loads = self.misses + self.prefetch_success
        return self.total_load_time / total_loads if total_loads > 0 else 0.0

    def __str__(self) -> str:
        """Format metrics as string."""
        return (
            f"Cache Metrics:\n"
            f"  Hit Rate: {self.hit_rate:.1%} ({self.hits}/{self.hits + self.misses})\n"
            f"  Prefetch: {self.prefetch_success} success, "
            f"{self.prefetch_failure} failed, {self.prefetch_cancelled} cancelled\n"
            f"  Avg Load Time: {self.average_load_time:.3f}s"
        )


# =============================================================================
# Cache Manager
# =============================================================================

class CacheManager:
    """
    Unified cache manager for VVMViz data.

    This class manages a two-layer caching system:
    1. Dataset cache: Handled by @lru_cache decorator on load functions
    2. Frame cache: Manual LRU cache for computed frame bundles

    The frame cache stores complete data bundles including:
    - Main variable (computed)
    - Wind overlay (if enabled, computed)
    - Contour overlay (if enabled, computed)

    Thread Safety
    -------------
    All cache operations are protected by a lock to ensure thread safety
    during concurrent access and background prefetching.

    Parameters
    ----------
    max_size : int, optional
        Maximum number of frames to cache (default: 200)
    enable_prefetch : bool, optional
        Enable background prefetching (default: True)
    max_workers : int, optional
        Number of prefetch worker threads (default: 1)
    """

    def __init__(
        self,
        max_size: int = 200,
        enable_prefetch: bool = True,
        max_workers: int = 1
    ):
        self.max_size = max_size
        self.enable_prefetch = enable_prefetch

        # Frame cache storage
        self.frame_cache: Dict[Tuple, Dict[str, Any]] = {}
        self.frame_cache_keys: list = []  # LRU ordering (oldest to newest)

        # Thread safety
        self.cache_lock = threading.Lock()

        # Prefetch management
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="vvmviz_prefetch"
        ) if enable_prefetch else None
        self.current_prefetch_future: Optional[Future] = None

        # Metrics
        self.metrics = CacheMetrics()

        logger.info(
            f"CacheManager initialized: max_size={max_size}, "
            f"prefetch={enable_prefetch}, workers={max_workers}"
        )

    def get(self, key: Tuple) -> Optional[Dict[str, Any]]:
        """
        Retrieve frame bundle from cache.

        Parameters
        ----------
        key : tuple
            Cache key (var_name, t_range, z_range)

        Returns
        -------
        dict or None
            Cached frame bundle if found, None otherwise
            Bundle contains: {'main': DataArray, 'wind': tuple, 'contour': DataArray}
        """
        with self.cache_lock:
            if key in self.frame_cache:
                # Cache hit: move key to end (mark as recently used)
                self.frame_cache_keys.remove(key)
                self.frame_cache_keys.append(key)

                self.metrics.hits += 1
                logger.debug(f"Cache HIT: {key}")

                return self.frame_cache[key]
            else:
                # Cache miss
                self.metrics.misses += 1
                logger.debug(f"Cache MISS: {key}")
                return None

    def put(self, key: Tuple, bundle: Dict[str, Any]) -> None:
        """
        Store frame bundle in cache with LRU eviction.

        Parameters
        ----------
        key : tuple
            Cache key (var_name, t_range, z_range)
        bundle : dict
            Frame bundle to cache
            Expected keys: 'main', 'wind', 'contour', 't_range', 'z_range', 'main_var'
        """
        with self.cache_lock:
            # Update existing key (move to end)
            if key in self.frame_cache:
                self.frame_cache_keys.remove(key)

            # Evict oldest if cache is full
            elif len(self.frame_cache) >= self.max_size:
                oldest_key = self.frame_cache_keys.pop(0)
                del self.frame_cache[oldest_key]
                logger.debug(f"Cache EVICT: {oldest_key}")

            # Insert new entry at end (most recent)
            self.frame_cache[key] = bundle
            self.frame_cache_keys.append(key)

            logger.debug(f"Cache PUT: {key} (size: {len(self.frame_cache)}/{self.max_size})")

    def clear(self) -> None:
        """Clear all cached frames."""
        with self.cache_lock:
            self.frame_cache.clear()
            self.frame_cache_keys.clear()
            logger.info("Cache cleared")

    def size(self) -> int:
        """Get current cache size."""
        with self.cache_lock:
            return len(self.frame_cache)

    def prefetch_async(
        self,
        request: FrameRequest,
        load_func: Callable[[FrameRequest], Dict[str, Any]]
    ) -> Optional[Future]:
        """
        Asynchronously prefetch a frame in the background.

        This cancels any currently running prefetch and starts a new one.

        Parameters
        ----------
        request : FrameRequest
            Request specification for the frame to prefetch
        load_func : callable
            Function to load the frame bundle: load_func(request) -> bundle

        Returns
        -------
        Future or None
            Future object for the prefetch task, or None if prefetch disabled
        """
        if not self.enable_prefetch or self.executor is None:
            return None

        cache_key = request.cache_key()

        # Check if already cached
        with self.cache_lock:
            if cache_key in self.frame_cache:
                logger.debug(f"Prefetch skipped (already cached): {cache_key}")
                return None

        # Cancel previous prefetch if still running
        if self.current_prefetch_future and not self.current_prefetch_future.done():
            cancelled = self.current_prefetch_future.cancel()
            if cancelled:
                self.metrics.prefetch_cancelled += 1
                logger.debug("Previous prefetch cancelled")

        # Submit new prefetch task
        future = self.executor.submit(self._prefetch_worker, request, load_func, cache_key)
        self.current_prefetch_future = future

        logger.debug(f"Prefetch started: {cache_key}")
        return future

    def _prefetch_worker(
        self,
        request: FrameRequest,
        load_func: Callable,
        cache_key: Tuple
    ) -> None:
        """
        Worker function for background prefetching.

        Parameters
        ----------
        request : FrameRequest
            Frame request
        load_func : callable
            Load function
        cache_key : tuple
            Cache key
        """
        try:
            # Check again if already cached (might have been loaded in main thread)
            with self.cache_lock:
                if cache_key in self.frame_cache:
                    return

            # Load the frame bundle
            start_time = time()
            bundle = load_func(request)
            elapsed = time() - start_time

            # Store in cache
            self.put(cache_key, bundle)

            # Update metrics
            self.metrics.prefetch_success += 1
            self.metrics.total_load_time += elapsed

            logger.info(
                f"Prefetch SUCCESS: {request.var_name} t={request.t_range} "
                f"({elapsed:.3f}s)"
            )

        except Exception as e:
            self.metrics.prefetch_failure += 1
            logger.error(f"Prefetch FAILED: {cache_key} - {e}", exc_info=True)

    def get_metrics(self) -> CacheMetrics:
        """Get current cache metrics."""
        return self.metrics

    def shutdown(self) -> None:
        """Shutdown the cache manager and cleanup resources."""
        if self.executor:
            logger.info("Shutting down cache executor...")
            self.executor.shutdown(wait=True)
        self.clear()
        logger.info("CacheManager shutdown complete")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()


# =============================================================================
# Global Cache Instance
# =============================================================================

# Default cache manager instance
# Can be replaced with a custom instance if needed
_default_cache_manager: Optional[CacheManager] = None


def get_cache_manager(
    max_size: Optional[int] = None,
    enable_prefetch: bool = True
) -> CacheManager:
    """
    Get or create the global cache manager instance.

    Parameters
    ----------
    max_size : int, optional
        Maximum cache size (only used when creating new instance)
    enable_prefetch : bool, optional
        Enable prefetch (only used when creating new instance)

    Returns
    -------
    CacheManager
        Global cache manager instance
    """
    global _default_cache_manager

    if _default_cache_manager is None:
        from vvmviz.config import MAX_FRAME_CACHE_SIZE

        size = max_size if max_size is not None else MAX_FRAME_CACHE_SIZE
        _default_cache_manager = CacheManager(
            max_size=size,
            enable_prefetch=enable_prefetch
        )

    return _default_cache_manager
