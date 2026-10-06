"""Compare compiled automata with Python's re module by brute force.

For each pattern, every string up to one character longer than the longest match is
checked against ``re.fullmatch``. The alphabet holds the letters the patterns use, one
digit standing for all ten (every digit behaves the same in these patterns), and one
foreign character. The number of states is checked against the number of distinct residual
languages, which is the size of the minimal automaton (Myhill-Nerode).
"""

import itertools
import random
import re

import pytest

from tagsort.errors import PatternError
from tagsort.grammar import Automaton, compile_pattern

HANDWRITTEN = [
    "a",
    "ab",
    "a|b",
    "[ab]",
    "a{2}",
    "a{1,3}",
    "a{0,2}b",
    "(ab){1,2}",
    "(a|ab)(c|bca)",
    "a|ab|abc",
    "(a|b){2}c{0,1}",
    "((a|b){2}|c){1,2}",
    "[a-c]{1,2}(b|c){0,2}",
    "(a{0,1}b){2}",
    "(a(b|c){0,1}){1,3}",
    r"\d",
    r"[\da]",
    r"a\-b",
    r"[a\-]",
    "[-a]{2}",
    "ab|ab",
    "(a|a){2}",
]


def _language(pattern: str, alphabet: str, max_length: int) -> set[str]:
    regex = re.compile(pattern, re.ASCII)
    return {
        "".join(chars)
        for length in range(max_length + 1)
        for chars in itertools.product(alphabet, repeat=length)
        if regex.fullmatch("".join(chars))
    }


def _alphabet(pattern: str) -> str:
    return "-abc7z" if "-" in pattern else "abc7z"


def _residual_count(language: set[str]) -> int:
    prefixes = {word[:i] for word in language for i in range(len(word) + 1)}
    residuals = {
        frozenset(word[len(prefix) :] for word in language if word.startswith(prefix))
        for prefix in prefixes
    }
    return len(residuals)


def _check_against_re(automaton: Automaton) -> None:
    pattern = automaton.pattern
    alphabet = _alphabet(pattern)
    language = _language(pattern, alphabet, automaton.max_length + 1)
    assert language, pattern

    for length in range(automaton.max_length + 2):
        for chars in itertools.product(alphabet, repeat=length):
            text = "".join(chars)
            assert automaton.matches(text) == (text in language), (pattern, text)

    assert automaton.min_length == min(map(len, language))
    assert automaton.max_length == max(map(len, language))
    assert automaton.num_states == _residual_count(language), pattern

    # Constrained decoding relies on allowed() listing exactly the characters that keep
    # the prefix completable.
    prefixes = {word[:i] for word in language for i in range(len(word) + 1)}
    for prefix in prefixes:
        state: int | None = automaton.start
        for char in prefix:
            assert state is not None
            state = automaton.step(state, char)
        assert state is not None
        expected = {
            word[len(prefix)]
            for word in language
            if word.startswith(prefix) and len(word) > len(prefix)
        }
        assert automaton.allowed(state) & set(alphabet) == expected, (pattern, prefix)
        assert automaton.is_accepting(state) == (prefix in language)


@pytest.mark.parametrize("pattern", HANDWRITTEN)
def test_handwritten_pattern_matches_re(pattern: str) -> None:
    _check_against_re(compile_pattern(pattern))


def _random_pattern(rng: random.Random, depth: int = 0) -> str:
    """Build a random pattern from the subset over the characters a, b, c and digits."""
    kind = rng.choice(["atom"] * 3 + ["concat", "group"] * (depth < 3))
    if kind == "atom":
        atom = rng.choice(["a", "b", "c", "[ab]", "[a-c]", "[bc]", r"\d", r"[\da]"])
    elif kind == "concat":
        return "".join(_random_pattern(rng, depth + 1) for _ in range(rng.randint(2, 3)))
    else:
        options = [_random_pattern(rng, depth + 1) for _ in range(rng.randint(1, 3))]
        atom = "(" + "|".join(options) + ")"
    roll = rng.random()
    if roll < 0.15:
        return f"{atom}{{{rng.randint(1, 2)}}}"
    if roll < 0.3:
        low = rng.randint(0, 1)
        return f"{atom}{{{low},{rng.randint(max(low, 1), 2)}}}"
    return atom


def _random_patterns(count: int) -> list[str]:
    rng = random.Random(20261006)
    return [_random_pattern(rng) for _ in range(count)]


@pytest.mark.parametrize("pattern", _random_patterns(1500))
def test_random_pattern_matches_re(pattern: str) -> None:
    if re.fullmatch(pattern, "", re.ASCII):
        with pytest.raises(PatternError) as info:
            compile_pattern(pattern)
        assert info.value.code == "matches_empty"
        return
    automaton = compile_pattern(pattern)
    if automaton.max_length <= 4:
        _check_against_re(automaton)
    else:
        _sample_against_re(automaton, random.Random(pattern))


def _sample_against_re(automaton: Automaton, rng: random.Random) -> None:
    """Compare on random strings and on random walks through the automaton."""
    regex = re.compile(automaton.pattern, re.ASCII)
    alphabet = _alphabet(automaton.pattern)
    for _ in range(300):
        length = rng.randint(0, automaton.max_length + 1)
        text = "".join(rng.choice(alphabet) for _ in range(length))
        assert automaton.matches(text) == bool(regex.fullmatch(text)), text
    for _ in range(300):
        state, text = automaton.start, ""
        while automaton.allowed(state) and not (
            automaton.is_accepting(state) and rng.random() < 0.3
        ):
            char = rng.choice(sorted(automaton.allowed(state)))
            next_state = automaton.step(state, char)
            assert next_state is not None
            state, text = next_state, text + char
        assert automaton.is_accepting(state)
        assert regex.fullmatch(text), text


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("a|b", "[ab]"),
        ("ab|ac", "a[bc]"),
        (r"\d{2}", "[0-9][0-9]"),
        ("(a|ab)(c|bcd)", "ac|abcd|abc|abbcd"),
        ("a{1,3}", "a|aa|aaa"),
        ("(ab){2}", "abab"),
        (r"[\-a]", "(-|a)"),
    ],
)
def test_equivalent_patterns_compile_to_identical_automata(first: str, second: str) -> None:
    a, b = compile_pattern(first), compile_pattern(second)
    assert a.num_states == b.num_states
    for state in range(a.num_states):
        assert list(a.edges(state)) == list(b.edges(state))
        assert a.is_accepting(state) == b.is_accepting(state)
