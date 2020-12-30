#!/usr/bin/env python3

"""
Main module for computing regular expressions from a list of strings/regex.

The libary expand a list of regex to a finite set of words, then generate a new
regular expression using linear optimization to find the near-shortest
combination. The computed regex should match exactly the same words as input.

The Regen class is intended for external uses. Other classes in the libary are
for internal uses only.

### Examples:

    >>> from regen import Regen

    >>> wordlist = ['ABC', 'ABD', 'BBC', 'BBD']
    >>> regen = Regen(wordlist)
    >>> regen.to_regex()
    '[AB]B[CD]'

    >>> wordlist = ['[AB]B[CD]', 'XYZ']
    >>> regen = Regen(wordlist)

    >>> regen.to_words()
    ['ABC', 'ABD', 'BBC', 'BBD', 'XYZ']

    >>> regen.to_regex()
    '(XYZ|[AB]B[CD])'
    
    >>> regen.to_regex(omitOuterParen=True)
    'XYZ|[AB]B[CD]'

### Author: `David Pi`
"""

__all__ = ("Regen",)

import re
from collections import defaultdict
from functools import lru_cache
from itertools import chain, compress, filterfalse
from typing import (
    Dict,
    FrozenSet,
    Iterable,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

from ortools.sat.python.cp_model import FEASIBLE, OPTIMAL, CpModel, CpSolver, LinearExpr

_specials = frozenset("{}()[]|?*+")
_not_special = _specials.isdisjoint
_rangeChars = frozenset("0123456789,")
_repetitions = frozenset("*?+{")
_split_token = re.compile(r"[^\\]|\\.").findall
Token = Tuple[str]


class Parser:

    __slots__ = ("result", "hold", "charset", "index", "subParser", "token", "_input")
    _is_char_block = re.compile(r"\\?.|\[(\^?\])?([^]]|\\\])*\]").fullmatch

    def __init__(self) -> None:
        self.result = [[]]
        self.hold = []
        self.charset = []
        self.index = 0
        self.subParser = self.token = None

    def parse(self, _input: Union[str, List[str]]) -> Iterator[Token]:
        """Convert a regular expression to a tokenset."""

        try:
            token = _split_token(_input)
        except TypeError:
            if not isinstance(_input, list):
                raise TypeError(f"Wrong type feed to Parser: {type(_input)}")
            token = _input

        if _not_special(token):  # Not regex we can handle
            yield tuple(token)
            return

        if self.index != 0:
            raise RuntimeError("Parser is not idle.")

        self.token = token
        self._input = _input

        result = self.result
        hold_append = self.hold.append
        eat = self._eat
        eatsuffix = self._eat_suffix

        char = eat()
        while char:

            if char in _specials:

                if char == "|":
                    self._concat_hold()
                    yield from map(tuple, result)
                    result[:] = [[]]

                elif char == "[":
                    self._charsetStrategy()

                elif char == "(":
                    self._parenStrategy()

                else:
                    raise ValueError(f"Invalid character '{char}': '{self.string}'")

            else:
                suffix = eatsuffix()
                if not suffix:
                    hold_append(char)
                elif suffix == "?":
                    self._concat_hold()
                    result.extend([*x, char] for x in tuple(result))
                else:
                    hold_append(char + suffix)

            char = eat()

        self._concat_hold()
        yield from map(tuple, result)
        result[:] = [[]]
        self.index = 0
        self.token = self._input = None

    def _charsetStrategy(self):

        start = self.index - 1  # including "["
        result = self.result
        hold = self.hold
        eat = self._eat
        charset = self.charset

        char = eat()
        if char == "^":
            try:
                # search from 2 chars after "^", end = 1 char after "]": [^]C]D (C-D)
                end = self.token.index("]", start + 3) + 1
            except ValueError:
                raise ValueError(f"Bad character set: {self.string}")
            if "[" in self.token[start + 1 : end - 1]:
                raise ValueError(f"Nested character set: {self.string}")

            self.index = end
            char = self._join_slice(start, end)  # [..]
            charset.append(char)
        else:
            while char:
                if char in "]-[":
                    if char == "]":
                        if charset:
                            break
                    elif char == "-":
                        if self._peek() != "]":
                            try:
                                lo = charset.pop()
                                hi = eat()
                                char = f"[{lo}{char}{hi}]"
                            except IndexError:
                                # "-" is the first char in charset
                                pass
                    else:
                        raise ValueError(f"Nested character set: {self.string}")
                charset.append(char)
                char = eat()
            else:
                raise ValueError(f"Bad character set: {self.string}")

        suffix = self._eat_suffix()

        if not suffix:
            if len(charset) > 1:
                result[:] = ([*x, *hold, y] for x in result for y in charset)
                hold.clear()
            else:
                hold.extend(charset)
        elif suffix == "?":
            self._concat_hold()
            result.extend([*x, y] for x in tuple(result) for y in charset)
        else:
            end = self.index  # from "[" to the end of suffix
            char = self._join_slice(start, end)
            hold.append(char)
        charset.clear()

    def _parenStrategy(self):

        start = self.index  # 1 char after "("
        result = self.result
        hold = self.hold

        balance = 1
        for end, char in enumerate(self.token[start:], start):
            if char in "()":
                if char == "(":
                    balance += 1
                else:
                    balance -= 1
                    if balance == 0:
                        break
        else:
            raise ValueError(f"Unbalanced parenthesis: {self.string}")

        self.index = end + 1  # 1 char after ")"
        suffix = self._eat_suffix()
        if start == end:  # empty group
            return

        if self.subParser is None:
            self.subParser = Parser()
        subToken = self.subParser.parse(self.token[start:end])

        if not suffix:
            result[:] = ([*x, *hold, *y] for y in subToken for x in result)
            hold.clear()
        elif suffix == "?":
            self._concat_hold()
            result.extend(tuple([*x, *y] for y in subToken for x in result))
        else:
            substr = optimize(frozenset(subToken), omitOuterParen=True)
            hold.append(
                substr + suffix
                if self._is_char_block(substr)
                else f"({substr}){suffix}"
            )

    def _concat_hold(self):
        hold = self.hold
        if hold:
            for s in self.result:
                s.extend(hold)
            hold.clear()

    def _eat(self) -> Optional[str]:
        """Consume one character in the token list."""
        try:
            char = self.token[self.index]
            self.index += 1
            return char
        except IndexError:
            pass

    def _peek(self) -> Optional[str]:
        try:
            return self.token[self.index]
        except IndexError:
            pass

    def _eat_suffix(self) -> Optional[str]:
        char = self._peek()
        if char not in _repetitions:
            return

        suffixStart = self.index  # first char of suffix
        if char == "{":
            try:
                # +2 so "{}" will not be matched
                suffixEnd = self.token.index("}", suffixStart + 2)
                if not _rangeChars.issuperset(self.token[suffixStart + 1 : suffixEnd]):
                    raise ValueError
            except ValueError:
                raise ValueError(f"Bad range format: {self.string}")

            self.index = suffixEnd + 1  # 1 char after "}"
            char = self._peek()

        while char and char in "*?+":
            self.index += 1
            char = self._peek()

        suffixEnd = self.index  # first char after suffix
        if suffixStart < suffixEnd:
            return self._join_slice(suffixStart, suffixEnd)

    def _join_slice(self, start: int, stop: int) -> str:
        return "".join(self.token[start:stop])

    @property
    def string(self):
        return self._input if isinstance(self._input, str) else "".join(self._input)


@lru_cache(maxsize=4096)
def optimize(tokenSet: FrozenSet[Token], omitOuterParen: bool = False) -> str:
    """Compute an optimized regular expression from the given tokenset."""

    tokenSet = set(tokenSet)
    if () in tokenSet:
        quantifier = "?"
        tokenSet.remove(())
    else:
        quantifier = ""

    if any(map(_is_word, tokenSet)):
        return _wordStrategy(tokenSet, quantifier, omitOuterParen)

    try:
        return _charsetStrategy(tokenSet, quantifier)
    except KeyError:
        return ""


def _wordStrategy(tokenSet: Set[Token], quantifier: str, omitOuterParen: bool) -> str:

    tokenSetLength = len(tokenSet)
    if tokenSetLength == 1:
        string = "".join(*tokenSet)
        return f"({string}){quantifier}" if quantifier else string

    result = []
    prefix = defaultdict(set)
    suffix = defaultdict(set)
    mirror = defaultdict(list)
    segment = {}
    candidate = {}
    unvisitCand = set()
    frozenUnion = frozenset().union

    for token in tokenSet:
        left = {token[:i]: token[i:] for i in range(1, len(token))}
        right = {v: k for k, v in left.items()}
        left[token] = right[token] = ()
        segment[token] = left, right
        for k in left:
            prefix[k].add(token)
        for k in right:
            suffix[k].add(token)

    prefix = _filter_affix(prefix)
    suffix = _filter_affix(suffix)

    if quantifier:
        tokenSet.add(())
        tokenSetLength += 1
    factor = tokenSetLength + 1

    while prefix or suffix:

        for i, source in enumerate((prefix, suffix)):

            for k, v in source.items():
                mirror[frozenset(segment[j][i][k] for j in v)].append(k)

            for k, v in mirror.items():
                if tokenSet.issuperset(k):
                    key = k.union(*map(source.get, v))
                    v.append(())
                else:
                    key = frozenUnion(*map(source.get, v))

                if key not in candidate:
                    left = optimize(k)
                    right = optimize(frozenset(v))
                    string = (left + right) if i else (right + left)
                    length = len(key)
                    value = (
                        sum(map(len, chain.from_iterable(key)))
                        + length
                        + 2 * (length == tokenSetLength)
                        - len(string)
                        - 1
                    ) * factor - length
                    candidate[key] = (value, string) if value > 0 else None

                if candidate[key]:
                    unvisitCand.add(key)

            mirror.clear()

        if not unvisitCand:
            break

        for key in _optimize_group(unvisitCand, candidate):
            result.append(candidate[key][1])
            tokenSet.difference_update(key)
        if not tokenSet:
            break

        prefix = _filter_affix(prefix, tokenSet)
        suffix = _filter_affix(suffix, tokenSet)

    if quantifier:
        try:
            tokenSet.remove(())
        except KeyError:
            quantifier = ""

    if tokenSet:
        chars = set(filterfalse(_is_word, tokenSet))
        if len(chars) > 2:
            tokenSet.difference_update(chars)
            result.append(_charsetStrategy(chars))
        result.extend(map("".join, tokenSet))

    result.sort()
    string = "|".join(result)

    if len(result) > 1 and not omitOuterParen or quantifier:
        return f"({string}){quantifier}"
    return string


def _charsetStrategy(tokenSet: Set[Token], quantifier: str = "") -> str:

    if len(tokenSet) > 1:
        char = sorted(chain.from_iterable(tokenSet))

        if ("]",) in tokenSet:
            char.insert(0, char.pop(char.index("]")))
        if ("-",) in tokenSet:
            char.append(char.pop(char.index("-")))

        return f'[{"".join(char)}]{quantifier}'

    return f"{tokenSet.pop()[0]}{quantifier}"


@lru_cache(maxsize=512)
def _is_word(token: Token) -> bool:
    return sum(map(len, token)) > 1


def _filter_affix(
    d: Dict[Token, Set[Token]], intersect: Set[Token] = None
) -> Dict[Token, Set[Token]]:
    """Keep groups which divide the same words at max common subsequence,
    and remove single member groups.

    - Example: (AB: ABC, ABD), (A: ABC, ABD), (ABC: ABC): only the first
        item will be keeped.
    """
    if intersect is None:
        stream = d.items()
    else:
        stream = _intersect_affix(d, intersect)

    tmp = {}
    setdefault = tmp.setdefault
    for k, v in stream:
        if len(v) > 1:
            key = frozenset(v)
            val = len(k), k
            if setdefault(key, val) < val:
                tmp[key] = val

    return {k: d[k] for _, k in tmp.values()}


def _intersect_affix(
    d: Dict[Token, Set[Token]], r: Set[Token]
) -> Iterator[Tuple[Token, Set[Token]]]:
    for i in d.items():
        i[1].intersection_update(r)
        yield i


def _optimize_group(
    unvisited: Set[FrozenSet[Token]], candidate: Dict[FrozenSet[Token], Tuple[int, str]]
):
    """Divide candidates into groups that each group is internally connected
    with common members. Then for each group, find the best non-overlapping
    members to reach the maximum length reduction.

    - Yield: Optimal keys
    - The input set (1st arg) will be finally emptied.
    """
    if len(unvisited) == 1:
        yield unvisited.pop()
        return

    stack = []
    pool = {}
    pool_get = pool.get

    while unvisited:

        currentKey = unvisited.pop()
        if all(map(currentKey.isdisjoint, unvisited)):
            yield currentKey
            continue

        model = CpModel()
        AddImplication = model.AddImplication
        pool[currentKey] = model.NewBoolVar("0")
        stack.append(currentKey)
        index = 1

        while stack:
            currentKey = stack.pop()
            unvisited.discard(currentKey)
            currentVarNot = pool[currentKey].Not()

            if () in currentKey:
                currentKey = currentKey.difference(((),))

            for nextKey in filterfalse(currentKey.isdisjoint, unvisited):
                nextVar = pool_get(nextKey)
                if nextVar is None:
                    nextVar = pool[nextKey] = model.NewBoolVar(f"{index}")
                    index += 1
                    stack.append(nextKey)
                AddImplication(nextVar, currentVarNot)

        model.Maximize(
            LinearExpr.ScalProd(pool.values(), tuple(candidate[k][0] for k in pool))
        )

        solver = CpSolver()
        status = solver.Solve(model)
        if status != OPTIMAL and status != FEASIBLE:
            raise RuntimeError(
                f"CP-SAT Solver failed, status: {solver.StatusName(status)}"
            )

        yield from compress(pool, map(solver.BooleanValue, pool.values()))
        pool.clear()


class Regen:

    __slots__ = ("_tokens", "_cache")

    def __init__(self, wordlist: Iterable[str]) -> None:
        """Convert a list of words to an optimized regular expression, or vise versa.

        ### Args:
        - `wordlist`: an iterable of strings.

        ### Methods:
        - `to_words`: Extract the regular expressions to a list of corresponding words.
        - `to_regex`: Return an optimized regular expression matching all the words.

        ### Examples:
        >>> from regen import Regen

        >>> wordlist = ['ABC', 'ABD', 'BBC', 'BBD']
        >>> regen = Regen(wordlist)
        >>> regen.to_regex()
        '[AB]B[CD]'

        >>> wordlist = ['[AB]B[CD]', 'XYZ']
        >>> regen = Regen(wordlist)

        >>> regen.to_words()
        ['ABC', 'ABD', 'BBC', 'BBD', 'XYZ']

        >>> regen.to_regex()
        '(XYZ|[AB]B[CD])'

        >>> regen.to_regex(omitOuterParen=True)
        'XYZ|[AB]B[CD]'
        """

        if isinstance(wordlist, str):
            wordlist = (wordlist,)
        elif not isinstance(wordlist, Iterable):
            raise TypeError("Input should be a list of strings.")

        self._tokens = frozenset(chain.from_iterable(map(Parser().parse, wordlist)))
        self._cache = {}

    def to_words(self) -> List[str]:
        """Extract the regular expressions to a list of corresponding words."""
        return sorted(map("".join, self._tokens))

    def to_regex(self, omitOuterParen: bool = False) -> str:
        """Return an optimized regular expression matching all the words.

        :omitOuterParen: If True, the outmost parentheses (if any) will be omited.
        """
        if not isinstance(omitOuterParen, bool):
            raise TypeError("omitOuterParen should be bool.")

        regex = self._cache.get(omitOuterParen)
        if regex is None:
            regex = self._cache[omitOuterParen] = optimize(self._tokens, omitOuterParen)
        return regex

    def raise_for_verify(self):
        regex = next(iter(self._cache.values()), None)
        if regex is None:
            regex = self.to_regex()

        if self._tokens != Regen(regex)._tokens:
            raise ValueError(
                "Extraction from computed regex is different from that of original wordlist."
            )

        for i in filterfalse(re.compile(regex).fullmatch, self.to_words()):
            if _not_special(i):
                raise ValueError(
                    f"Computed regex does not fully match this word: '{i}'"
                )


def parse_arguments():

    import argparse

    parser = argparse.ArgumentParser(
        description="""Generate regular expressions from a set of words and regexes.
Author: David Pi <libertypi@gmail.com>""",
        epilog="""examples:
  %(prog)s cat bat at "fat|boat"
  %(prog)s -e "[AB]C[DE]"
  %(prog)s -f words.txt""",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "-e",
        "--extract",
        dest="mode",
        action="store_const",
        const="extract",
        help="extract Regex to a list of corresponding words",
    )
    mode_group.add_argument(
        "-c",
        "--compute",
        dest="mode",
        action="store_const",
        const="compute",
        help="compute an optimized regex matching the words (default)",
    )
    parser.set_defaults(mode="compute")

    parser.add_argument(
        "-v",
        "--verify",
        dest="verify",
        action="store_true",
        help="verify the generated regex by rematching it against input",
    )
    parser.add_argument(
        "-o",
        "--omit-paren",
        dest="omit",
        action="store_true",
        help="omit the outer parentheses, if any",
    )

    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "-f",
        "--file",
        dest="file",
        type=argparse.FileType("r"),
        help="take text from FILE, one word per line",
    )
    target_group.add_argument(
        dest="word",
        nargs="*",
        action="store",
        default=(),
        help="a list of words/regexes, one word per argument",
    )

    return parser.parse_args()


def main():

    args = parse_arguments()

    if args.file is None:
        regen = Regen(args.word)
    else:
        regen = Regen(filter(None, args.file.read().splitlines()))
        args.file.close()

    if args.mode == "extract":
        for word in regen.to_words():
            print(word)
    else:
        regex = regen.to_regex(omitOuterParen=args.omit)
        print(regex)

        if args.verify:
            print("\nLength:", len(regex))
            try:
                regen.raise_for_verify()
            except ValueError as e:
                print("Verify failed:", e)
            else:
                print("Verify passed.")


if __name__ == "__main__":
    main()
