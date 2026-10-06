"""Parser for the tag pattern regex subset.

The subset is specified in ``docs/patterns.md``. Every pattern describes a finite set of
non-empty strings, so it compiles to an acyclic automaton. Anything outside the subset is
rejected with a :class:`~tagsort.errors.PatternError` that points at the offending
character and carries a stable error code.
"""

from __future__ import annotations

import string
from dataclasses import dataclass

from tagsort.errors import PatternError

__all__ = [
    "MAX_CLASS_SIZE",
    "MAX_DEPTH",
    "MAX_LENGTH",
    "MAX_REPEAT",
    "Alternation",
    "CharSet",
    "Concat",
    "Node",
    "Repeat",
    "max_length",
    "min_length",
    "parse",
]

MAX_REPEAT = 64
"""Largest upper bound allowed in a ``{n,m}`` quantifier."""

MAX_LENGTH = 256
"""Longest string a pattern may match."""

MAX_CLASS_SIZE = 512
"""Largest number of characters a single character class may contain."""

MAX_DEPTH = 16
"""Deepest nesting of groups allowed."""

DIGITS = frozenset(string.digits)

# Characters that may follow a backslash to stand for themselves.
_ESCAPABLE = frozenset(string.punctuation) | {" "}


@dataclass(frozen=True)
class CharSet:
    """Exactly one character taken from ``chars``."""

    chars: frozenset[str]


@dataclass(frozen=True)
class Concat:
    """Each item in order."""

    items: tuple[Node, ...]


@dataclass(frozen=True)
class Alternation:
    """Exactly one of the options."""

    options: tuple[Node, ...]


@dataclass(frozen=True)
class Repeat:
    """``node`` repeated between ``low`` and ``high`` times, inclusive."""

    node: Node
    low: int
    high: int


Node = CharSet | Concat | Alternation | Repeat


def parse(pattern: str) -> Node:
    """Parse ``pattern`` into a syntax tree.

    Raises:
        PatternError: If the pattern is outside the supported subset.
    """
    if not isinstance(pattern, str):
        raise TypeError(f"pattern must be a str, not {type(pattern).__name__}")
    node = _Parser(pattern).parse()
    if min_length(node) == 0:
        raise PatternError(
            "pattern matches the empty string; at least one character must be required",
            code="matches_empty",
            pattern=pattern,
            position=0,
        )
    longest = max_length(node)
    if longest > MAX_LENGTH:
        raise PatternError(
            f"pattern matches strings of up to {longest} characters; the maximum is {MAX_LENGTH}",
            code="too_long",
            pattern=pattern,
            position=0,
        )
    return node


def max_length(node: Node) -> int:
    """Return the length of the longest string matched by ``node``."""
    if isinstance(node, CharSet):
        return 1
    if isinstance(node, Concat):
        return sum(max_length(item) for item in node.items)
    if isinstance(node, Alternation):
        return max(max_length(option) for option in node.options)
    return node.high * max_length(node.node)


def min_length(node: Node) -> int:
    """Return the length of the shortest string matched by ``node``."""
    if isinstance(node, CharSet):
        return 1
    if isinstance(node, Concat):
        return sum(min_length(item) for item in node.items)
    if isinstance(node, Alternation):
        return min(min_length(option) for option in node.options)
    return node.low * min_length(node.node)


def _is_control(code: int) -> bool:
    return code < 0x20 or 0x7F <= code < 0xA0


class _Parser:
    """Recursive-descent parser over the pattern string."""

    def __init__(self, pattern: str) -> None:
        self.pattern = pattern
        self.pos = 0
        self.depth = 0  # number of groups currently open

    # Helpers

    def _error(self, code: str, message: str, position: int | None = None) -> PatternError:
        return PatternError(
            message,
            code=code,
            pattern=self.pattern,
            position=self.pos if position is None else position,
        )

    def _peek(self) -> str | None:
        return self.pattern[self.pos] if self.pos < len(self.pattern) else None

    def _literal(self, char: str) -> CharSet:
        if _is_control(ord(char)):
            raise self._error(
                "control_character", f"control character U+{ord(char):04X} is not allowed"
            )
        self.pos += 1
        return CharSet(frozenset(char))

    # Grammar

    def parse(self) -> Node:
        if not self.pattern:
            raise self._error("empty_pattern", "pattern is empty")
        node = self._alternation()
        # _alternation stops only at the end, or at a ')' that _sequence has rejected.
        assert self._peek() is None
        return node

    def _alternation(self) -> Node:
        options = [self._sequence()]
        while self._peek() == "|":
            self.pos += 1
            options.append(self._sequence())
        return options[0] if len(options) == 1 else Alternation(tuple(options))

    def _sequence(self) -> Node:
        items: list[Node] = []
        while (char := self._peek()) is not None and char not in "|)":
            items.append(self._quantified())
        if char == ")" and self.depth == 0:
            raise self._error("unbalanced_paren", "unbalanced ')': no group is open")
        if not items:
            raise self._error(
                "empty_alternative", "empty alternative; use {0,1} to make a part optional"
            )
        return items[0] if len(items) == 1 else Concat(tuple(items))

    def _quantified(self) -> Node:
        node = self._atom()
        if self._peek() == "{":
            node = self._quantifier(node)
            if self._peek() in {"{", "*", "+", "?"}:
                raise self._error(
                    "invalid_quantifier", "a quantifier cannot follow another quantifier"
                )
        return node

    def _atom(self) -> Node:
        char = self._peek()
        assert char is not None
        if char == "(":
            return self._group()
        if char == "[":
            return self._class()
        if char == "\\":
            return self._escape(in_class=False)
        if char in {"*", "+"}:
            raise self._error(
                "unbounded_quantifier",
                f"quantifier '{char}' is not supported: patterns must have a bounded length; "
                "use {n} or {n,m}",
            )
        if char == "?":
            raise self._error(
                "unsupported_quantifier", "quantifier '?' is not supported; use {0,1}"
            )
        if char == "{":
            raise self._error("invalid_quantifier", "quantifier without anything to repeat")
        if char in {"^", "$"}:
            raise self._error(
                "anchor",
                f"anchor '{char}' is not supported; patterns always match the whole tag text",
            )
        if char == ".":
            raise self._error(
                "wildcard",
                "wildcard '.' is not supported; list the allowed characters in a class "
                "such as [A-Z0-9]",
            )
        if char in {"]", "}"}:
            raise self._error(
                "unescaped_special", f"unescaped '{char}'; write \\{char} for a literal '{char}'"
            )
        return self._literal(char)

    def _group(self) -> Node:
        open_pos = self.pos
        self.pos += 1
        if self._peek() == "?":
            raise self._error(
                "group_extension",
                "'(?' extensions such as non-capturing groups are not supported",
            )
        if self._peek() == ")":
            raise self._error("empty_group", "empty group")
        if self.depth == MAX_DEPTH:
            raise self._error(
                "nested_too_deep", f"groups are nested more than {MAX_DEPTH} levels deep", open_pos
            )
        self.depth += 1
        node = self._alternation()
        self.depth -= 1
        if self._peek() != ")":
            raise self._error(
                "unbalanced_paren", "unbalanced '(': the group is never closed", open_pos
            )
        self.pos += 1
        return node

    def _escape(self, *, in_class: bool) -> CharSet:
        start = self.pos
        self.pos += 1
        char = self._peek()
        if char is None:
            raise self._error("unsupported_escape", "pattern ends with a lone backslash", start)
        self.pos += 1
        if char == "d":
            return CharSet(DIGITS)
        if char in _ESCAPABLE:
            return CharSet(frozenset(char))
        if char.isdigit():
            raise self._error("backreference", "backreferences are not supported", start)
        where = " inside a character class" if in_class else ""
        raise self._error(
            "unsupported_escape",
            f"escape '\\{char}' is not supported{where}; only \\d and escaped "
            "punctuation such as \\. or \\- are allowed",
            start,
        )

    def _class(self) -> CharSet:
        open_pos = self.pos
        self.pos += 1
        if self._peek() == "^":
            raise self._error(
                "negated_class", "negated character classes '[^...]' are not supported"
            )
        chars: set[str] = set()
        while (char := self._peek()) != "]":
            if char is None:
                raise self._error(
                    "unbalanced_bracket",
                    "unbalanced '[': the character class is never closed",
                    open_pos,
                )
            low_pos = self.pos
            low = self._class_member()
            is_range = (
                self._peek() == "-"
                and self.pos + 1 < len(self.pattern)
                and self.pattern[self.pos + 1] != "]"
            )
            if is_range:
                self.pos += 1
                high_pos = self.pos
                high = self._class_member()
                chars.update(self._range(low, high, low_pos, high_pos))
            else:
                chars |= low.chars
            if len(chars) > MAX_CLASS_SIZE:
                raise self._error(
                    "class_too_large",
                    f"character class has more than {MAX_CLASS_SIZE} characters",
                    open_pos,
                )
        if not chars:
            raise self._error("empty_class", "empty character class")
        self.pos += 1
        return CharSet(frozenset(chars))

    def _range(self, low: CharSet, high: CharSet, low_pos: int, high_pos: int) -> set[str]:
        if len(low.chars) != 1 or len(high.chars) != 1:
            bad = low_pos if len(low.chars) != 1 else high_pos
            raise self._error("invalid_range", r"\d cannot be a range endpoint", bad)
        (first,) = low.chars
        (last,) = high.chars
        if ord(first) > ord(last):
            raise self._error("invalid_range", f"range '{first}-{last}' is out of order", low_pos)
        if ord(last) - ord(first) + 1 > MAX_CLASS_SIZE:
            raise self._error(
                "class_too_large",
                f"range '{first}-{last}' has more than {MAX_CLASS_SIZE} characters",
                low_pos,
            )
        if any(_is_control(code) for code in range(ord(first), ord(last) + 1)):
            raise self._error(
                "control_character", f"range '{first}-{last}' contains control characters", low_pos
            )
        return {chr(code) for code in range(ord(first), ord(last) + 1)}

    def _class_member(self) -> CharSet:
        char = self._peek()
        assert char is not None
        if char == "\\":
            return self._escape(in_class=True)
        if char == "[":
            raise self._error(
                "unescaped_special", "unescaped '[' inside a character class; write \\["
            )
        return self._literal(char)

    def _quantifier(self, node: Node) -> Repeat:
        open_pos = self.pos
        close = self.pattern.find("}", self.pos)
        if close == -1:
            raise self._error(
                "unbalanced_brace", "unbalanced '{': the quantifier is never closed", open_pos
            )
        body = self.pattern[self.pos + 1 : close]
        low_text, comma, high_text = body.partition(",")
        if comma and low_text and not high_text and _is_number(low_text):
            raise self._error(
                "unbounded_quantifier",
                "unbounded quantifier {n,} is not supported; give an upper bound {n,m}",
                open_pos,
            )
        if not _is_number(low_text) or (comma and not _is_number(high_text)):
            raise self._error(
                "invalid_quantifier", "quantifier must be {n} or {n,m} with integers", open_pos
            )
        low = int(low_text)
        high = int(high_text) if comma else low
        if low > high:
            raise self._error(
                "invalid_quantifier", f"quantifier {{{body}}} has its bounds out of order", open_pos
            )
        if high == 0:
            raise self._error("invalid_quantifier", "quantifier repeats zero times", open_pos)
        if high > MAX_REPEAT:
            raise self._error(
                "repeat_too_large", f"quantifier upper bound is above {MAX_REPEAT}", open_pos
            )
        self.pos = close + 1
        return Repeat(node, low, high)


def _is_number(text: str) -> bool:
    """Return whether ``text`` is a non-empty run of ASCII digits."""
    return text.isascii() and text.isdigit()
