# Tag patterns

Each tag in a profile has a `pattern`: a regular expression written in a small subset of the
usual syntax. A pattern describes every text a tag of that kind may carry. TagSort compiles it
to a finite automaton and uses it to:

- reject a profile with a clear error when the pattern is invalid;
- check that a reading is plausible;
- restrict the characters allowed at each position during recognition (constrained decoding).

The subset is small so that it can be implemented identically in any language. Test vectors
for ports are in [`reference/grammar.v1.json`](../reference/grammar.v1.json).

## Semantics

- A pattern always matches the **whole** tag text. There are no anchors.
- Matching is **case-sensitive**. Write `[Aa]` to accept both cases.
- A pattern must describe a **finite** set of **non-empty** strings.
- Lengths and positions are counted in **Unicode code points**, not bytes or UTF-16 units.
- **A gap is not a character.** When reading, a space where the pattern allows none is
  ignored, and the tag is reported without it: `GJ\d{5}` reads a tag written `GJ 07966`
  as `GJ07966`, as surely as one written without the gap. Where the pattern has a space
  (`NMC \d{5}`), the space is required there. Other characters, such as a dash, always
  count.

## Syntax

| Syntax | Meaning | Example |
| --- | --- | --- |
| `c` | The character `c` itself, for any printable character that is not special | `MD` |
| `\d` | One ASCII digit, `0` to `9` only (not other Unicode digits) | `\d{5}` |
| `\p` | The punctuation character `p` itself, for any ASCII punctuation or a space | `\.`, `\-`, `\(` |
| `[...]` | One character from the class; members are characters, escapes and ranges `a-z` | `[A-Z0-9]` |
| `x{n}` | `x` exactly `n` times | `\d{4}` |
| `x{n,m}` | `x` between `n` and `m` times, inclusive | `\d{3,4}` |
| `(a\|b)` | One of the alternatives; groups can be nested and repeated | `(MD\|NHM)\d{5}` |
| `a\|b` | Alternation at the top level of the pattern | `MD\d{5}\|NHM\d{4}` |

Special characters, which must be escaped with a backslash to be literal:
`\ ^ $ . | ? * + ( ) [ ] { }`.

Inside a class:

- `-` is literal when it is the first or last member, or escaped as `\-`;
- `^` is literal unless it comes first (negated classes are not supported);
- `]`, `[` and `\` must be escaped;
- `\d` may be a member but not a range endpoint.

A quantifier applies to the single atom before it: a character, an escape, a class or a group.

## Limits

| Limit | Value |
| --- | --- |
| Upper bound `m` of a quantifier | 64 |
| Longest matched string | 256 characters |
| Characters in one class | 512 |
| Group nesting depth | 16 |
| States of the compiled automaton | 10,000 |

## Not supported

| Feature | Use instead |
| --- | --- |
| `*`, `+`, `{n,}` | `{n,m}` with an upper bound |
| `?` | `{0,1}` |
| `.` | a class listing the allowed characters, such as `[A-Z0-9]` |
| `^`, `$` | nothing: patterns always match the whole text |
| `[^...]` | a class listing the allowed characters |
| `\w`, `\s`, `\b` and other letter escapes | an explicit class |
| `(?:...)`, lookarounds, named groups | a plain group `(...)` |
| backreferences `\1` | none |
| control characters (U+0000 to U+001F, U+007F to U+009F) | none |

## Error codes

An invalid pattern raises `PatternError` with a human-readable message, the position of the
offending character, and one of these stable codes. Applications can use the code to show
their own message.

| Code | Cause |
| --- | --- |
| `empty_pattern` | The pattern is the empty string |
| `matches_empty` | The pattern can match the empty string, for example `\d{0,3}` |
| `too_long` | The pattern can match a string longer than 256 characters |
| `too_complex` | The compiled automaton would exceed 10,000 states |
| `unbalanced_paren` | A `(` is never closed, or a `)` has no matching `(` |
| `unbalanced_bracket` | A `[` is never closed |
| `unbalanced_brace` | A `{` is never closed |
| `empty_group` | `()` |
| `empty_alternative` | An alternative is empty, as in `(a\|)` or `\|a` |
| `empty_class` | `[]` |
| `nested_too_deep` | Groups are nested more than 16 levels deep |
| `unbounded_quantifier` | `*`, `+` or `{n,}` |
| `unsupported_quantifier` | `?` |
| `invalid_quantifier` | A malformed quantifier: nothing to repeat, `{a}`, bounds out of order, `{0}`, or two quantifiers in a row |
| `repeat_too_large` | A quantifier upper bound above 64 |
| `anchor` | `^` or `$` outside a class |
| `wildcard` | `.` |
| `unescaped_special` | A `]` or `}` outside a class, or a `[` inside one |
| `group_extension` | `(?` |
| `unsupported_escape` | A backslash followed by a letter other than `d`, or a lone trailing backslash |
| `backreference` | A backslash followed by a digit |
| `negated_class` | `[^...]` |
| `invalid_range` | A range out of order such as `[z-a]`, or `\d` used as a range endpoint |
| `class_too_large` | A class or range with more than 512 characters |
| `control_character` | A control character, alone or inside a range |

## Examples

| Pattern | Matches | Does not match |
| --- | --- | --- |
| `MD\d{5}` | `MD04127` | `MD4127`, `md04127` |
| `[A-Z]{2}-\d{3,4}` | `AB-123`, `AB-1234` | `AB123`, `A-123` |
| `(MD\|NHM)\d{5}` | `MD04127`, `NHM04127` | `NH04127` |
| `[A-Z]{1,4} \d{1,6}(\.\d{1,2}){0,1}` | `MNHN 2001`, `MNHN 2001.12` | `MNHN 2001.`, `MNHN2001` |
