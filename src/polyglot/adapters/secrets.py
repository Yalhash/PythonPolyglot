"""API keys in the OS keyring (Windows Credential Manager, macOS Keychain)."""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

SERVICE = "polyglot"


class KeyringSecretStore:
    def __init__(self, service: str = SERVICE) -> None:
        self._service = service

    def get(self, name: str) -> str | None:
        try:
            import keyring
            from keyring.errors import KeyringError
        except ImportError:
            log.warning("keyring is not installed; secrets cannot be stored")
            return None
        try:
            value = keyring.get_password(self._service, name)
        except KeyringError:
            log.warning("could not read %s from the OS keyring", name, exc_info=True)
            return None
        return value or None

    def set(self, name: str, value: str | None) -> None:
        import keyring
        from keyring.errors import PasswordDeleteError

        if value:
            keyring.set_password(self._service, name, value)
            return
        try:
            keyring.delete_password(self._service, name)
        except PasswordDeleteError:
            pass  # nothing stored


class MemorySecretStore:
    """In-memory store for tests and for running without a keyring."""

    def __init__(self, values: dict[str, str] | None = None) -> None:
        self.values = dict(values or {})

    def get(self, name: str) -> str | None:
        return self.values.get(name) or None

    def set(self, name: str, value: str | None) -> None:
        if value:
            self.values[name] = value
        else:
            self.values.pop(name, None)
