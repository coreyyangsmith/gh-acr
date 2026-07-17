"""Backend factory re-exports (compatibility shims for handlers)."""

from .openai_backend import create_openai_backend
from .groq_backend import create_groq_backend
from .local_backend import create_local_backend

__all__ = [
    "create_openai_backend",
    "create_groq_backend",
    "create_local_backend",
]
