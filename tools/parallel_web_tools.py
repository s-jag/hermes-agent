#!/usr/bin/env python3
"""
Parallel Web Tools — function library for web_tools.py

This module implements web search and content extraction using the
Parallel API (https://parallel.ai). It is NOT a standalone tool module;
tool registration is handled by web_tools.py, which delegates to these
functions when the web_search_backend is "parallel" (or "auto" with
PARALLEL_API_KEY set).

Functions:
- parallel_search_tool: Search the web (via Parallel Search API)
- parallel_extract_tool: Extract content from web pages (via Parallel Extract API)

LLM Processing:
- Reuses the same LLM content processing pipeline from web_tools.py
- Large pages are chunked, summarized, and synthesized automatically

Debug Mode:
- Set PARALLEL_TOOLS_DEBUG=true to enable detailed logging
- Creates parallel_tools_debug_UUID.json in ./logs directory

Usage:
    from tools.parallel_web_tools import parallel_search_tool, parallel_extract_tool

    # Search the web
    results = parallel_search_tool("Python machine learning libraries", limit=5)

    # Extract content from URLs
    content = await parallel_extract_tool(["https://example.com"], format="markdown")
"""

import asyncio
import json
import logging
from typing import Any, Dict, List

from parallel import AsyncParallel, Parallel

from tools.debug_helpers import DebugSession
from tools.parallel_config import (
    get_parallel_api_key,
    get_search_mode,
)
from tools.web_tools import clean_base64_images, process_content_with_llm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Client initialization (lazy, cached)
# ---------------------------------------------------------------------------

_parallel_client = None
_async_parallel_client = None


def _get_parallel_client():
    """Get or create the sync Parallel client (lazy initialization)."""
    global _parallel_client
    if _parallel_client is None:
        api_key = get_parallel_api_key()
        if not api_key:
            raise ValueError("PARALLEL_API_KEY environment variable not set")
        _parallel_client = Parallel(api_key=api_key)
    return _parallel_client


def _get_async_parallel_client():
    """Get or create the async Parallel client (lazy initialization)."""
    global _async_parallel_client
    if _async_parallel_client is None:
        api_key = get_parallel_api_key()
        if not api_key:
            raise ValueError("PARALLEL_API_KEY environment variable not set")
        _async_parallel_client = AsyncParallel(api_key=api_key)
    return _async_parallel_client


def check_parallel_api_key() -> bool:
    """Check if the Parallel API key is available in environment variables."""
    return bool(get_parallel_api_key())


# ---------------------------------------------------------------------------
# Debug session
# ---------------------------------------------------------------------------

_debug = DebugSession("parallel_tools", env_var="PARALLEL_TOOLS_DEBUG")


# ---------------------------------------------------------------------------
# Search tool (synchronous)
# ---------------------------------------------------------------------------

def parallel_search_tool(query: str, limit: int = 5) -> str:
    """
    Search the web using the Parallel Search API.

    Returns results in the same JSON format as web_search_tool() for
    seamless backend switching.

    Args:
        query: The search query to look up
        limit: Maximum number of results to return (default: 5)

    Returns:
        JSON string: {"success": true, "data": {"web": [...]}} on success,
                     {"error": "message"} on failure.
    """
    debug_call_data = {
        "parameters": {"query": query, "limit": limit},
        "error": None,
        "results_count": 0,
        "original_response_size": 0,
        "final_response_size": 0,
    }

    try:
        from tools.interrupt import is_interrupted

        if is_interrupted():
            return json.dumps({"error": "Interrupted", "success": False})

        mode = get_search_mode()
        logger.info(
            "Searching the web for: '%s' (limit: %d, mode: %s)", query, limit, mode
        )

        client = _get_parallel_client()
        response = client.beta.search(
            search_queries=[query],
            mode=mode,
            max_results=limit,
            excerpts={"max_chars_per_result": 10000},
        )

        # Map Parallel response to Firecrawl-compatible format
        web_results = []
        results_list = response.results if hasattr(response, "results") else []

        for idx, result in enumerate(results_list):
            title = getattr(result, "title", None) or ""
            url = getattr(result, "url", "") or ""
            excerpts = getattr(result, "excerpts", None) or []
            publish_date = getattr(result, "publish_date", None)

            # Map excerpts to description for compatibility
            if excerpts:
                description = "\n\n".join(excerpts)
            else:
                description = ""

            entry = {
                "title": title,
                "url": url,
                "description": description,
                "position": idx + 1,
            }
            if publish_date:
                entry["publish_date"] = publish_date

            web_results.append(entry)

        results_count = len(web_results)
        logger.info("Found %d search results", results_count)

        response_data = {"success": True, "data": {"web": web_results}}

        debug_call_data["results_count"] = results_count

        result_json = json.dumps(response_data, indent=2, ensure_ascii=False)

        debug_call_data["final_response_size"] = len(result_json)
        _debug.log_call("parallel_search_tool", debug_call_data)
        _debug.save()

        return result_json

    except Exception as e:
        error_msg = f"Error searching web: {str(e)}"
        logger.debug("%s", error_msg)

        debug_call_data["error"] = error_msg
        _debug.log_call("parallel_search_tool", debug_call_data)
        _debug.save()

        return json.dumps({"error": error_msg}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Extract tool (asynchronous)
# ---------------------------------------------------------------------------

async def parallel_extract_tool(
    urls: List[str],
    format: str = None,
    use_llm_processing: bool = True,
) -> str:
    """
    Extract content from web pages using the Parallel Extract API.

    Returns results in the same JSON format as web_extract_tool() for
    seamless backend switching.

    Args:
        urls: List of URLs to extract content from
        format: Desired output format (ignored; Parallel always returns markdown)
        use_llm_processing: Whether to process content with LLM for summarization

    Returns:
        JSON string: {"results": [{"title", "content", "error"}]}
    """
    debug_call_data = {
        "parameters": {
            "urls": urls,
            "format": format,
            "use_llm_processing": use_llm_processing,
        },
        "error": None,
        "pages_extracted": 0,
        "pages_processed_with_llm": 0,
        "original_response_size": 0,
        "final_response_size": 0,
        "compression_metrics": [],
        "processing_applied": [],
    }

    try:
        from tools.interrupt import is_interrupted as _is_interrupted

        logger.info("Extracting content from %d URL(s) via Parallel", len(urls))

        client = _get_async_parallel_client()

        # Call Parallel Extract API with all URLs at once
        try:
            response = await client.beta.extract(
                urls=urls,
                full_content=True,
            )
        except Exception as api_err:
            logger.debug("Parallel extract API call failed: %s", api_err)
            raise

        # Collect results
        results: List[Dict[str, Any]] = []
        api_results = response.results if hasattr(response, "results") else []
        api_errors = response.errors if hasattr(response, "errors") else []

        # Build a set of URLs that had errors
        error_urls: Dict[str, str] = {}
        for err in api_errors:
            err_url = getattr(err, "url", None) or ""
            err_msg = getattr(err, "error", None) or str(err)
            error_urls[err_url] = err_msg

        for result in api_results:
            url = getattr(result, "url", "") or ""
            title = getattr(result, "title", "") or ""
            full_content = getattr(result, "full_content", None) or ""
            excerpts = getattr(result, "excerpts", None) or []

            # Prefer full_content, fall back to joined excerpts
            content = full_content or "\n\n".join(excerpts)

            results.append({
                "url": url,
                "title": title,
                "content": content,
                "raw_content": content,
            })

        # Add error entries for URLs that failed
        result_urls = {r["url"] for r in results}
        for url in urls:
            if url not in result_urls:
                err_msg = error_urls.get(url, "Extraction failed")
                results.append({
                    "url": url,
                    "title": "",
                    "content": "",
                    "raw_content": "",
                    "error": err_msg,
                })

        pages_extracted = len([r for r in results if r.get("raw_content")])
        logger.info("Extracted content from %d pages", pages_extracted)

        debug_call_data["pages_extracted"] = pages_extracted
        debug_call_data["original_response_size"] = sum(
            len(r.get("raw_content", "")) for r in results
        )

        # LLM processing for large content
        if use_llm_processing:
            logger.info("Processing extracted content with LLM (parallel)...")
            debug_call_data["processing_applied"].append("llm_processing")

            async def process_single_result(result):
                if _is_interrupted():
                    return result, None, "interrupted"

                url = result.get("url", "Unknown URL")
                title = result.get("title", "")
                raw_content = result.get("raw_content", "")

                if not raw_content:
                    return result, None, "no_content"

                original_size = len(raw_content)

                processed = await process_content_with_llm(
                    raw_content, url, title
                )

                if processed:
                    processed_size = len(processed)
                    compression_ratio = (
                        processed_size / original_size if original_size > 0 else 1.0
                    )
                    result["content"] = processed
                    metrics = {
                        "url": url,
                        "original_size": original_size,
                        "processed_size": processed_size,
                        "compression_ratio": compression_ratio,
                    }
                    return result, metrics, "processed"
                else:
                    metrics = {
                        "url": url,
                        "original_size": original_size,
                        "processed_size": original_size,
                        "compression_ratio": 1.0,
                        "reason": "content_too_short",
                    }
                    return result, metrics, "too_short"

            tasks = [process_single_result(r) for r in results]
            processed_results = await asyncio.gather(*tasks)

            for result, metrics, status in processed_results:
                url = result.get("url", "Unknown URL")
                if status == "processed":
                    debug_call_data["compression_metrics"].append(metrics)
                    debug_call_data["pages_processed_with_llm"] += 1
                    logger.info("%s (processed)", url)
                elif status == "too_short":
                    if metrics:
                        debug_call_data["compression_metrics"].append(metrics)
                    logger.info("%s (no processing - content too short)", url)
                else:
                    logger.info("%s (%s)", url, status)

        # Trim output to minimal fields
        trimmed_results = [
            {
                "title": r.get("title", ""),
                "content": r.get("content", ""),
                "error": r.get("error"),
            }
            for r in results
        ]
        trimmed_response = {"results": trimmed_results}

        if not any(r.get("content") for r in trimmed_results):
            result_json = json.dumps(
                {"error": "Content was inaccessible or not found"}, ensure_ascii=False
            )
        else:
            result_json = json.dumps(trimmed_response, indent=2, ensure_ascii=False)

        cleaned_result = clean_base64_images(result_json)

        debug_call_data["final_response_size"] = len(cleaned_result)
        debug_call_data["processing_applied"].append("base64_image_removal")

        _debug.log_call("parallel_extract_tool", debug_call_data)
        _debug.save()

        return cleaned_result

    except Exception as e:
        error_msg = f"Error extracting content: {str(e)}"
        logger.debug("%s", error_msg)

        debug_call_data["error"] = error_msg
        _debug.log_call("parallel_extract_tool", debug_call_data)
        _debug.save()

        return json.dumps({"error": error_msg}, ensure_ascii=False)
