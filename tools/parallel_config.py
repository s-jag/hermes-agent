"""Parallel API configuration utilities.

Import-safe: only depends on os and hermes_cli.config.load_config (lazy).
No circular dependencies with tool modules.
"""

import os
import logging

logger = logging.getLogger(__name__)

# Defaults
DEFAULT_SEARCH_MODE = "fast"
DEFAULT_TASK_PROCESSOR = "pro"
VALID_SEARCH_MODES = ("ultra-fast", "fast", "one-shot", "agentic")
VALID_TASK_PROCESSORS = ("lite", "base", "core", "pro", "ultra")


def get_parallel_api_key():
    """Return the Parallel API key, or None if not set."""
    return os.getenv("PARALLEL_API_KEY") or None


def is_parallel_available():
    """Check if Parallel API key is configured."""
    return bool(get_parallel_api_key())


def get_search_mode():
    """Return the Parallel search mode (ultra-fast, fast, one-shot, agentic)."""
    mode = os.getenv("PARALLEL_SEARCH_MODE", DEFAULT_SEARCH_MODE).lower()
    if mode not in VALID_SEARCH_MODES:
        logger.warning(
            "Invalid PARALLEL_SEARCH_MODE '%s', using '%s'", mode, DEFAULT_SEARCH_MODE
        )
        return DEFAULT_SEARCH_MODE
    return mode


def get_task_processor():
    """Return the Parallel Task API processor tier (lite, base, core, pro, ultra)."""
    proc = os.getenv("PARALLEL_TASK_PROCESSOR", DEFAULT_TASK_PROCESSOR).lower()
    if proc not in VALID_TASK_PROCESSORS:
        logger.warning(
            "Invalid PARALLEL_TASK_PROCESSOR '%s', using '%s'",
            proc,
            DEFAULT_TASK_PROCESSOR,
        )
        return DEFAULT_TASK_PROCESSOR
    return proc


def get_web_search_backend():
    """Return the active web search backend: 'parallel', 'firecrawl', or 'none'.

    Reads ``web_search_backend`` from config.yaml (default: ``'auto'``).

    * ``'auto'``     -- prefer Parallel if PARALLEL_API_KEY is set, else Firecrawl
    * ``'parallel'`` -- always use Parallel
    * ``'firecrawl'``-- always use Firecrawl
    """
    try:
        from hermes_cli.config import load_config

        config = load_config()
        backend = config.get("web_search_backend", "auto")
    except Exception:
        backend = os.getenv("WEB_SEARCH_BACKEND", "auto")

    backend = str(backend).lower().strip()

    if backend == "parallel":
        return "parallel"
    elif backend == "firecrawl":
        return "firecrawl"
    elif backend == "auto":
        if is_parallel_available():
            return "parallel"
        elif os.getenv("FIRECRAWL_API_KEY"):
            return "firecrawl"
        return "none"
    else:
        logger.warning("Unknown web_search_backend '%s', using auto", backend)
        if is_parallel_available():
            return "parallel"
        elif os.getenv("FIRECRAWL_API_KEY"):
            return "firecrawl"
        return "none"
