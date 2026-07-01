"""Typed exceptions for the SCR client.

The reference clients call ``sys.exit`` on socket/protocol failures; this library
raises instead, so callers (the smoke script now, the Gymnasium env in T-004)
get a clear, catchable signal. See docs/design/scr-client.md §8.
"""

from __future__ import annotations


class ScrError(Exception):
    """Base class for all SCR client errors."""


class ScrConnectionError(ScrError):
    """Handshake never completed (retry budget exhausted without ***identified***)."""


class ScrProtocolError(ScrError):
    """A datagram could not be parsed (no sensor groups / unrecoverable shape)."""
