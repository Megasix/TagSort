"""Tag pattern grammar: a regex subset compiled to a finite automaton.

Internal package. The subset is specified in ``docs/patterns.md``.
"""

from tagsort.grammar.automaton import Automaton, compile_pattern

__all__ = ["Automaton", "compile_pattern"]
