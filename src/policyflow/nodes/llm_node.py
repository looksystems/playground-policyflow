"""Base class for LLM-based nodes with caching and rate limiting."""

import hashlib
import time
from pathlib import Path
from typing import Optional
from threading import Lock

import yaml
from pocketflow import Node

from ..config import WorkflowConfig
from ..llm import call_llm as _call_llm


class LLMNode(Node):
    """Base class for LLM-based nodes with caching and throttling."""

    # Class-level default model - subclasses should override
    default_model: str = "anthropic/claude-sonnet-4-20250514"

    # Class-level locks for thread-safe cache and rate limiting
    _cache_lock = Lock()
    _rate_limit_lock = Lock()
    # Class-level rate limiting state (per-instance tracking)
    _rate_limit_tokens = {}  # instance_id -> (tokens, last_update)

    def __init__(
        self,
        config: WorkflowConfig,
        model: str | None = None,
        cache_ttl: int = 3600,
        rate_limit: int = None,
    ):
        """
        Initialize LLM node with caching and rate limiting.

        Args:
            config: Workflow configuration
            model: LLM model identifier (uses class default_model if not provided)
            cache_ttl: Cache time-to-live in seconds, 0 = disabled
            rate_limit: Requests per minute, None = unlimited
        """
        super().__init__(max_retries=config.max_retries)
        self.config = config
        self.model = model if model is not None else self.default_model
        self.cache_ttl = cache_ttl  # seconds, 0 = disabled
        self.rate_limit = rate_limit  # requests per minute, None = unlimited
        self._instance_id = id(self)  # Unique ID for rate limiting

        # Initialize cache directory
        self._cache_dir = Path(".cache")
        if self.cache_ttl > 0:
            self._cache_dir.mkdir(exist_ok=True)

        # Initialize rate limiting tokens for this instance
        if self.rate_limit is not None:
            with self._rate_limit_lock:
                self._rate_limit_tokens[self._instance_id] = (
                    float(self.rate_limit),
                    time.time(),
                )

    def _cache_key(self, prompt: str) -> str:
        """
        Generate cache key from prompt hash.

        Args:
            prompt: The prompt string to hash

        Returns:
            SHA256 hash of the prompt
        """
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    def _check_cache(self, key: str) -> Optional[dict]:
        """
        Return cached result or None.

        Args:
            key: Cache key from _cache_key()

        Returns:
            Cached result dict or None if not found/expired
        """
        if self.cache_ttl == 0:
            return None

        cache_file = self._cache_dir / f"{key}.yaml"

        with self._cache_lock:
            if not cache_file.exists():
                return None

            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached_data = yaml.safe_load(f)

                # Check TTL
                if cached_data and "timestamp" in cached_data:
                    age = time.time() - cached_data["timestamp"]
                    if age < self.cache_ttl:
                        return cached_data.get("result")

                # Cache expired, remove file
                cache_file.unlink(missing_ok=True)
            except (yaml.YAMLError, OSError, KeyError):
                # Corrupted cache, remove it
                cache_file.unlink(missing_ok=True)

        return None

    def _store_cache(self, key: str, result: dict):
        """
        Store result in cache.

        Args:
            key: Cache key from _cache_key()
            result: Result dictionary to cache
        """
        if self.cache_ttl == 0:
            return

        cache_file = self._cache_dir / f"{key}.yaml"
        cache_data = {"timestamp": time.time(), "result": result}

        with self._cache_lock:
            try:
                with open(cache_file, "w", encoding="utf-8") as f:
                    yaml.safe_dump(cache_data, f, default_flow_style=False)
            except OSError:
                # Fail silently if cache write fails
                pass

    def _check_rate_limit(self) -> bool:
        """
        Check and update rate limit using token bucket algorithm.
        Waits if rate limit is exceeded.

        Returns:
            True (always returns after waiting if needed)
        """
        if self.rate_limit is None:
            return True

        with self._rate_limit_lock:
            # Get current token count and last update time
            tokens, last_update = self._rate_limit_tokens.get(
                self._instance_id, (float(self.rate_limit), time.time())
            )

            now = time.time()
            time_passed = now - last_update

            # Refill tokens based on time passed (tokens per second = rate_limit / 60)
            refill_rate = self.rate_limit / 60.0
            tokens = min(self.rate_limit, tokens + (time_passed * refill_rate))

            # If we have tokens, consume one
            if tokens >= 1.0:
                tokens -= 1.0
                self._rate_limit_tokens[self._instance_id] = (tokens, now)
                return True

            # Need to wait for tokens
            wait_time = (1.0 - tokens) / refill_rate

        # Wait outside the lock
        time.sleep(wait_time)

        # Update tokens after waiting
        with self._rate_limit_lock:
            self._rate_limit_tokens[self._instance_id] = (0.0, time.time())

        return True

    def call_llm(
        self,
        prompt: str,
        system_prompt: str | None = None,
        yaml_response: bool = True,
        span_name: str | None = None,
    ) -> dict:
        """
        Call LLM with caching and throttling. Used by subclasses in exec().

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            yaml_response: Whether to parse response as YAML
            span_name: Optional name for the trace span (for observability)

        Returns:
            LLM response as dict (if yaml_response=True) or string
        """
        # Generate cache key from both prompts
        cache_input = f"{system_prompt or ''}\n{prompt}"
        cache_key = self._cache_key(cache_input)

        # Check cache first
        cached_result = self._check_cache(cache_key)
        if cached_result is not None:
            return cached_result

        # Check rate limit (will wait if needed)
        self._check_rate_limit()

        # Call LLM
        result = _call_llm(
            prompt=prompt,
            system_prompt=system_prompt,
            model=self.model,
            config=self.config,
            yaml_response=yaml_response,
            span_name=span_name,
        )

        # Store in cache
        if yaml_response and isinstance(result, dict):
            self._store_cache(cache_key, result)

        return result
