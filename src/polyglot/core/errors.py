"""Typed errors that are part of each port's contract.

Adapters translate vendor/library exceptions into these. The core never sees raw
``urllib``/``onnxruntime``/``ctranslate2`` exceptions.
"""

from __future__ import annotations

from polyglot.core.models import ErrorKind


class PolyglotError(Exception):
    kind: ErrorKind = ErrorKind.INTERNAL
    retryable: bool = False


class ProviderError(PolyglotError):
    """A cloud/remote provider failed."""

    def __init__(
        self, message: str, *, kind: ErrorKind | None = None, retryable: bool | None = None
    ) -> None:
        super().__init__(message)
        if kind is not None:
            self.kind = kind
        if retryable is not None:
            self.retryable = retryable


class ProviderTimeout(ProviderError):
    kind = ErrorKind.TIMEOUT
    retryable = True


class ProviderUnavailable(ProviderError):
    kind = ErrorKind.UNAVAILABLE
    retryable = True


class QuotaExceeded(ProviderError):
    kind = ErrorKind.QUOTA
    retryable = False


class AuthError(ProviderError):
    kind = ErrorKind.AUTH
    retryable = False


class BadResponse(ProviderError):
    kind = ErrorKind.BAD_RESPONSE
    retryable = False


class TranscriptionFailed(PolyglotError):
    kind = ErrorKind.TRANSCRIPTION


class AudioSourceError(PolyglotError):
    kind = ErrorKind.AUDIO


class ConfigError(PolyglotError):
    kind = ErrorKind.NOT_CONFIGURED
