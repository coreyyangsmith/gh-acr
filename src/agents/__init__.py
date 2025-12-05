"""Agent package for LLM-powered merge conflict resolution.

This package provides the core agent implementations for resolving Git merge
conflicts using Large Language Models. It includes multiple resolution
strategies ranging from simple baselines to sophisticated multi-agent workflows.

Package Structure
-----------------
- **agent/**: Single-turn LLM-based resolver (simple prompt → merged output)
- **base_agent/**: Deterministic baselines (always select Parent A or B)
- **bypass/**: Multi-agent with bypass analyzer (summarize → analyze → plan → patch → review)
- **bypass_only/**: Lightweight analyzer-only (no merge, just parent selection)
- **bypass7/**: Multi-agent with tuned prompts
- **multi_agent/**: Consolidated implementation shared by bypass variants
- **backends/**: LLM backend implementations (OpenAI, Groq, local)

Key Modules
-----------
- **llm_base**: Centralized LLM backend registry (get_backend function)
- **state**: TypedDict definitions for pipeline state
- **utils**: Shared utilities (template rendering, text extraction)
- **callbacks**: LangChain callbacks for rate limiting and cost tracking
- **token_utils**: Token counting utilities

Resolution Strategies
---------------------
1. **base_a/base_b**: Deterministic baselines
   - Always select Parent A or Parent B content
   - Useful as comparison baselines

2. **agent**: Single-turn LLM resolver
   - One prompt with diffs → merged output
   - Fast but less sophisticated

3. **bypass**: Multi-agent pipeline
   - Summarizer: Describe what each parent changed
   - Analyzer: Decide if bypass (A/B) or merge
   - Planner: Create per-file merge strategy
   - Resolver: Apply the plan to produce merged code
   - Reviewer: Check quality with retry loop

4. **bypass_only**: Lightweight classification
   - Only summarize and analyze
   - Select parent without LLM merge
   - Fast for cases where one parent is clearly better

Example Usage
-------------
>>> from src.agents.llm_base import get_backend
>>> from src.agents.bypass import resolve_conflict_bypass_multi_agent_node
>>> 
>>> # Get an LLM backend
>>> encoder, llm = get_backend("openai/gpt-4o-mini")
>>> 
>>> # Run the bypass multi-agent resolver
>>> state = {"model_name": "openai/gpt-4o-mini", ...}
>>> result = resolve_conflict_bypass_multi_agent_node(state)
>>> merged = result["resolved_contents"]
"""

from __future__ import annotations

from .utils import render_template, extract_text_content, scenario_file_list

__all__ = [
    "render_template",
    "extract_text_content",
    "scenario_file_list",
]
