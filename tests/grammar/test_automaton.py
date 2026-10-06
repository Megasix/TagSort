import threading

import pytest

from tagsort.errors import PatternError, ProfileError, TagSortError
from tagsort.grammar import automaton as automaton_module
from tagsort.grammar import compile_pattern


def test_walk_through_a_pattern() -> None:
    automaton = compile_pattern(r"MD\d{2}")
    assert automaton.allowed(automaton.start) == {"M"}
    state = automaton.step(automaton.start, "M")
    assert state is not None
    state = automaton.step(state, "D")
    assert state is not None
    assert automaton.allowed(state) == set("0123456789")
    assert not automaton.is_accepting(state)
    assert automaton.step(state, "X") is None


def test_alphabet_and_edges() -> None:
    automaton = compile_pattern("b|a[xy]")
    assert automaton.alphabet == {"a", "b", "x", "y"}
    assert [char for char, _ in automaton.edges(automaton.start)] == ["a", "b"]


def test_matches_rejects_unknown_characters_and_prefixes() -> None:
    automaton = compile_pattern("abc")
    assert automaton.matches("abc")
    assert not automaton.matches("ab")
    assert not automaton.matches("abcd")
    assert not automaton.matches("aXc")
    assert not automaton.matches("")


def test_repr() -> None:
    assert repr(compile_pattern("ab")) == "Automaton('ab', states=3)"


def test_compilation_is_cached() -> None:
    assert compile_pattern("a{3}") is compile_pattern("a{3}")


def test_automaton_is_immutable() -> None:
    automaton = compile_pattern("ab")
    with pytest.raises(TypeError):
        automaton._transitions[0]["z"] = 1  # type: ignore[index]


def test_shared_between_threads() -> None:
    automaton = compile_pattern(r"[A-Z]{2}\d{3}")
    results: list[bool] = []

    def worker() -> None:
        results.extend(automaton.matches(f"AB{n:03d}") for n in range(1000))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert results == [True] * 8000


def test_too_complex(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(automaton_module, "MAX_STATES", 5)
    with pytest.raises(PatternError) as info:
        compile_pattern.__wrapped__("abcdef")
    assert info.value.code == "too_complex"


def test_pattern_must_be_a_string() -> None:
    with pytest.raises(TypeError, match="must be a str"):
        compile_pattern(42)


def test_large_class_compiles_quickly() -> None:
    automaton = compile_pattern("[Ā-˿]{64}")
    assert automaton.num_states == 65
    assert len(automaton.allowed(0)) == 512


def test_pattern_error_message_points_at_the_problem() -> None:
    with pytest.raises(PatternError) as info:
        compile_pattern(r"MD\d+")
    error = info.value
    assert isinstance(error, ProfileError)
    assert isinstance(error, TagSortError)
    assert isinstance(error, ValueError)
    assert error.location is None
    assert str(error) == (
        "quantifier '+' is not supported: patterns must have a bounded length; "
        "use {n} or {n,m}\n"
        "  MD\\d+\n"
        "      ^"
    )


def test_pattern_error_can_be_located_in_a_profile() -> None:
    with pytest.raises(PatternError) as info:
        compile_pattern("a?")
    located = info.value.at("tags[1].pattern")
    assert located.location == "tags[1].pattern"
    assert located.code == "unsupported_quantifier"
    assert located.position == 1
    assert located.message == info.value.message
    assert str(located).startswith("tags[1].pattern: quantifier '?' is not supported")


def test_profile_error_without_location() -> None:
    error = ProfileError("profile is empty")
    assert str(error) == "profile is empty"
    assert error.location is None
