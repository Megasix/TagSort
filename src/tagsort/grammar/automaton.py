"""Compile a parsed pattern to a minimal deterministic finite automaton.

Patterns describe finite languages, so the automaton is acyclic. It is built in three
steps: Thompson construction of an NFA, subset construction of a DFA, then minimization
by merging states with identical futures. States are numbered in breadth-first order
from the start state, visiting characters in code point order, so the numbering is
canonical: two patterns that match the same strings yield identical automata.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator, Mapping
from functools import lru_cache
from types import MappingProxyType

from tagsort.errors import PatternError
from tagsort.grammar.parser import (
    Alternation,
    CharSet,
    Concat,
    Node,
    Repeat,
    max_length,
    min_length,
    parse,
)

__all__ = ["MAX_STATES", "Automaton", "compile_pattern"]

MAX_STATES = 10_000
"""Largest number of DFA states a pattern may compile to."""


class Automaton:
    """Minimal DFA accepting exactly the strings a pattern matches.

    State ``start`` is the initial state. A missing transition means the character is
    not allowed at that point. Every state can reach an accepting state.

    Instances are immutable and safe to share between threads.
    """

    __slots__ = ("_accepting", "_allowed", "_transitions", "max_length", "min_length", "pattern")

    start = 0

    def __init__(
        self,
        pattern: str,
        transitions: tuple[Mapping[str, int], ...],
        accepting: frozenset[int],
        *,
        min_length: int,
        max_length: int,
    ) -> None:
        """Wrap a DFA built by :func:`compile_pattern`; not meant to be called directly."""
        self.pattern = pattern
        self._transitions = transitions
        self._accepting = accepting
        self._allowed = tuple(frozenset(edges) for edges in transitions)
        self.min_length = min_length
        self.max_length = max_length

    def __repr__(self) -> str:
        """Return a short description with the pattern and the number of states."""
        return f"Automaton({self.pattern!r}, states={self.num_states})"

    @property
    def num_states(self) -> int:
        """Number of states."""
        return len(self._transitions)

    @property
    def alphabet(self) -> frozenset[str]:
        """Every character that appears on some transition."""
        return frozenset().union(*self._allowed)

    def step(self, state: int, char: str) -> int | None:
        """Return the state reached from ``state`` by reading ``char``, or ``None``."""
        return self._transitions[state].get(char)

    def allowed(self, state: int) -> frozenset[str]:
        """Return the characters that may be read from ``state``."""
        return self._allowed[state]

    def is_accepting(self, state: int) -> bool:
        """Return whether the text read so far is a complete match."""
        return state in self._accepting

    def matches(self, text: str) -> bool:
        """Return whether the whole of ``text`` matches the pattern."""
        state: int | None = self.start
        for char in text:
            assert state is not None
            state = self.step(state, char)
            if state is None:
                return False
        assert state is not None
        return self.is_accepting(state)

    def edges(self, state: int) -> Iterator[tuple[str, int]]:
        """Yield ``(char, next_state)`` for each transition, in code point order."""
        transitions = self._transitions[state]
        for char in sorted(transitions):
            yield char, transitions[char]


@lru_cache(maxsize=256)
def compile_pattern(pattern: str) -> Automaton:
    """Compile ``pattern`` to its minimal automaton.

    Raises:
        PatternError: If the pattern is outside the supported subset or too complex.
    """
    node = parse(pattern)
    nfa = _Nfa()
    nfa.final = nfa.build(node, nfa.new_state())
    transitions, accepting = _determinize(nfa, pattern)
    transitions, accepting = _minimize(transitions, accepting)
    return Automaton(
        pattern,
        tuple(MappingProxyType(edges) for edges in transitions),
        frozenset(accepting),
        min_length=min_length(node),
        max_length=max_length(node),
    )


class _Nfa:
    """Thompson NFA with character-set and epsilon transitions."""

    def __init__(self) -> None:
        self.edges: list[list[tuple[frozenset[str], int]]] = []
        self.epsilon: list[list[int]] = []
        self.final = -1

    def new_state(self) -> int:
        self.edges.append([])
        self.epsilon.append([])
        return len(self.edges) - 1

    def build(self, node: Node, start: int) -> int:
        """Add ``node`` starting at state ``start``; return its end state."""
        if isinstance(node, CharSet):
            end = self.new_state()
            self.edges[start].append((node.chars, end))
            return end
        if isinstance(node, Concat):
            state = start
            for item in node.items:
                state = self.build(item, state)
            return state
        if isinstance(node, Alternation):
            end = self.new_state()
            for option in node.options:
                self.epsilon[self.build(option, start)].append(end)
            return end
        return self._build_repeat(node, start)

    def _build_repeat(self, node: Repeat, start: int) -> int:
        state = start
        for _ in range(node.low):
            state = self.build(node.node, state)
        end = self.new_state()
        self.epsilon[state].append(end)
        for _ in range(node.high - node.low):
            state = self.build(node.node, state)
            self.epsilon[state].append(end)
        return end

    def closure(self, states: frozenset[int]) -> frozenset[int]:
        seen = set(states)
        stack = list(states)
        while stack:
            for target in self.epsilon[stack.pop()]:
                if target not in seen:
                    seen.add(target)
                    stack.append(target)
        return frozenset(seen)


def _determinize(nfa: _Nfa, pattern: str) -> tuple[list[dict[str, int]], set[int]]:
    start = nfa.closure(frozenset({0}))
    index = {start: 0}
    order = [start]
    transitions: list[dict[str, int]] = []
    accepting: set[int] = set()
    for current, subset in enumerate(order):
        if nfa.final in subset:
            accepting.add(current)
        moves: dict[str, set[int]] = {}
        for state in subset:
            for chars, target in nfa.edges[state]:
                for char in chars:
                    moves.setdefault(char, set()).add(target)
        edges: dict[str, int] = {}
        for char, targets in moves.items():
            closed = nfa.closure(frozenset(targets))
            if closed not in index:
                if len(order) == MAX_STATES:
                    raise PatternError(
                        f"pattern is too complex: it needs more than {MAX_STATES} states",
                        code="too_complex",
                        pattern=pattern,
                        position=0,
                    )
                index[closed] = len(order)
                order.append(closed)
            edges[char] = index[closed]
        transitions.append(edges)
    return transitions, accepting


def _minimize(
    transitions: list[dict[str, int]], accepting: set[int]
) -> tuple[list[dict[str, int]], set[int]]:
    """Merge equivalent states of an acyclic DFA and renumber them canonically."""
    # Visit states in post-order so every successor has its class before its predecessors.
    post_order: list[int] = []
    visited = {0}
    stack: list[tuple[int, Iterator[int]]] = [(0, iter(transitions[0].values()))]
    while stack:
        state, successors = stack[-1]
        for successor in successors:
            if successor not in visited:
                visited.add(successor)
                stack.append((successor, iter(transitions[successor].values())))
                break
        else:
            stack.pop()
            post_order.append(state)

    # Two states are equivalent when they agree on acceptance and on the class reached by
    # every character. Classes are keyed by that signature.
    signature_class: dict[tuple[bool, tuple[tuple[str, int], ...]], int] = {}
    class_of: dict[int, int] = {}
    class_edges: list[dict[str, int]] = []
    class_accepting: list[bool] = []
    for state in post_order:
        edges = {char: class_of[target] for char, target in transitions[state].items()}
        signature = (state in accepting, tuple(sorted(edges.items())))
        if signature not in signature_class:
            signature_class[signature] = len(class_edges)
            class_edges.append(edges)
            class_accepting.append(state in accepting)
        class_of[state] = signature_class[signature]

    # Renumber in breadth-first order from the start state.
    start = class_of[0]
    number = {start: 0}
    queue = deque([start])
    result: list[dict[str, int]] = []
    result_accepting: set[int] = set()
    while queue:
        cls = queue.popleft()
        if class_accepting[cls]:
            result_accepting.add(number[cls])
        edges = {}
        for char in sorted(class_edges[cls]):
            target = class_edges[cls][char]
            if target not in number:
                number[target] = len(number)
                queue.append(target)
            edges[char] = number[target]
        result.append(edges)
    return result, result_accepting
