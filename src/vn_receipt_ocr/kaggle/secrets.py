from __future__ import annotations

import os


def _client_factory() -> object | None:
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore
        return UserSecretsClient()
    except Exception:
        return None


def get_secret_or_none(key: str) -> str | None:
    """Try environment variable first, then Kaggle Secrets, else None."""
    val = os.environ.get(key)
    if val:
        return val
    client = _client_factory()
    if client is None:
        return None
    try:
        return client.get_secret(key)  # type: ignore[attr-defined]
    except Exception:
        return None
