"""Langfuse client factory — points at the self-hosted instance from .env."""
import os

from langfuse import Langfuse


def get_langfuse_client() -> Langfuse:
    """Return a configured Langfuse client.

    Reads LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, and LANGFUSE_HOST from the
    environment.  LANGFUSE_HOST defaults to http://localhost:3000 so local
    docker-compose works with no extra config.
    """
    host = os.environ.get("LANGFUSE_HOST", "http://localhost:3000")
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY", "")
    return Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        host=host,
    )
