"""Typed exceptions raised by TagSort."""

from __future__ import annotations

__all__ = ["ImageError", "PatternError", "ProfileError", "ProviderError", "TagSortError"]


class TagSortError(Exception):
    """Base class for every error raised by TagSort on purpose."""


class ProfileError(TagSortError, ValueError):
    """A profile is malformed or does not conform to ``profile.v1.json``.

    Attributes:
        location: Where the problem is in the profile document, for example
            ``tags[1].pattern``, or ``None`` when it concerns the whole document.
    """

    def __init__(self, message: str, *, location: str | None = None) -> None:
        """Create the error.

        Args:
            message: What is wrong, in one sentence.
            location: Where the problem is in the profile document, if known.
        """
        self.message = message
        self.location = location
        super().__init__(f"{location}: {message}" if location else message)


class PatternError(ProfileError):
    """A tag pattern is outside the supported regex subset.

    Attributes:
        code: Stable identifier of the problem, such as ``unbounded_quantifier``, for
            applications that show their own messages. The codes are listed in
            ``docs/patterns.md``.
        pattern: The offending pattern.
        position: Index of the offending character in ``pattern``, counted in Unicode
            code points.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str,
        pattern: str,
        position: int,
        location: str | None = None,
    ) -> None:
        """Create the error.

        Args:
            message: What is wrong, in one sentence.
            code: Stable identifier of the problem.
            pattern: The offending pattern.
            position: Index of the offending character in ``pattern``.
            location: Where the pattern is in the profile document, if known.
        """
        self.code = code
        self.pattern = pattern
        self.position = position
        caret = f"\n  {pattern}\n  {' ' * position}^"
        super().__init__(message + caret, location=location)
        self.message = message

    def at(self, location: str) -> PatternError:
        """Return a copy of this error located at ``location`` in a profile."""
        return PatternError(
            self.message,
            code=self.code,
            pattern=self.pattern,
            position=self.position,
            location=location,
        )


class ImageError(TagSortError, ValueError):
    """An image cannot be decoded, or its format is not supported."""


class ProviderError(TagSortError):
    """A vision API provider failed to return a usable answer.

    Attributes:
        provider: Name of the provider, for example ``anthropic``.
        reason: Stable identifier of the failure: ``authentication``, ``quota`` (no
            credit left on the account), ``bad_request``, ``rate_limit``, ``server``,
            ``network``, ``refused``, ``truncated`` or ``invalid_response``.
        status_code: HTTP status code, when the provider answered with an error.
        retryable: Whether retrying the same request later may succeed.
    """

    _RETRYABLE = frozenset({"rate_limit", "server", "network"})

    def __init__(
        self, message: str, *, provider: str, reason: str, status_code: int | None = None
    ) -> None:
        """Create the error.

        Args:
            message: What went wrong, in one sentence. Never contains the API key.
            provider: Name of the provider.
            reason: Stable identifier of the failure.
            status_code: HTTP status code, if any.
        """
        self.provider = provider
        self.reason = reason
        self.status_code = status_code
        self.retryable = reason in self._RETRYABLE
        super().__init__(f"{provider}: {message}")
