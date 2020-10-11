#!/usr/bin/env python3

"""
Main module for computing regular expressions from a list of strings/regex.

The libary expand a list of regex to a finite set of words, then generate a new
regular expression using linear optimization to find the near-shortest combination.
The computed regex should match exactly the same words as input.

The Regen class is intended for external uses, example:

from regenerator import Regen

wordlist = ['ABC', 'ABD', 'BBC', 'BBD']
regen = Regen(wordlist)
regen.to_regex() -> '[AB]B[CD]'

wordlist = ['[AB]B[CD]', 'XYZ']
regen = Regen(wordlist)
regen.to_text() -> ['ABC', 'ABD', 'BBC', 'BBD', 'XYZ']
regen.to_regex() -> '(XYZ|[AB]B[CD])'
regen.to_regex(omitOuterParen=True) -> 'XYZ|[AB]B[CD]'

Other classes in the libary are for internal uses only.

Author: David Pi
"""

__all__ = ["Regen"]

import re
from collections import defaultdict, deque
from functools import lru_cache
from itertools import chain, filterfalse

from ortools.linear_solver import pywraplp

_specials = frozenset("{}()[]|?*+")


class Parser:

    _rangeChars = frozenset("0123456789,")
    _suffixes = frozenset("*?+{")
    _repetitions = frozenset("*?+")
    _psplit = re.compile(r"[^\\]|\\.").findall
    _is_char_block = re.compile(r"\\?.|\[(\^?\])?([^]]|\\\])*\]").fullmatch

    def __init__(self) -> None:
        self.result = [[]]
        self.hold = []
        self.charset = []
        self.index = 0
        self.subParser = self.optimizer = None

    def parse(self, string):
        """Convert a regular expression to a tokenset."""

        try:
            token = self._psplit(string)
        except TypeError:
            if not isinstance(string, list):
                raise TypeError(f"Wrong type feed to Parser: {string}")
            token = string

        if _specials.isdisjoint(token):  # Not regex we can handle
            yield tuple(token)
            return

        self.token = token
        self._string = string

        hold = self.hold
        eat = self._eat
        eatsuffix = self._eat_suffix

        char = eat()
        while char:

            if char == "|":
                self._concat_hold()
                yield from map(tuple, self.result)
                self._reset_result()

            elif char == "[":
                self._charsetStrategy()

            elif char == "(":
                self._parenStrategy()

            elif char in _specials:
                raise ValueError(f"Invalid character '{char}': '{self.string}'")

            else:
                suffix = eatsuffix()
                if not suffix:
                    hold.append(char)
                elif suffix == "?":
                    self._concat_hold()
                    self.result.extend(tuple([*x, char] for x in self.result))
                else:
                    hold.append(f"{char}{suffix}")

            char = eat()

        self._concat_hold()
        yield from map(tuple, self.result)
        self._reset_result()
        self.index = 0
        self.token = self._string = None

    def _charsetStrategy(self):
        start = self.index - 1  # including "["
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
                if char == "[":
                    raise ValueError(f"Nested character set: {self.string}")
                if char == "]" and charset:
                    break
                elif char == "-" and self._peek() != "]":
                    try:
                        lo = charset.pop()
                        hi = eat()
                        char = f"[{lo}{char}{hi}]"
                    except IndexError:  # "-" is the first char in charset
                        pass
                charset.append(char)
                char = eat()
            else:
                raise ValueError(f"Bad character set: {self.string}")

        suffix = self._eat_suffix()

        if not suffix:
            if len(charset) > 1:
                self.result = [[*x, *hold, y] for x in self.result for y in charset]
                hold.clear()
            else:
                hold.extend(charset)
        elif suffix == "?":
            self._concat_hold()
            self.result.extend(tuple([*x, y] for x in self.result for y in charset))
        else:
            end = self.index  # from "[" to the end of suffix
            char = self._join_slice(start, end)
            hold.append(char)
        charset.clear()

    def _parenStrategy(self):

        start = self.index  # 1 char after "("
        hold = self.hold

        balance = 1
        for end, char in enumerate(self.token[start:], start):
            if char == "(":
                balance += 1
            elif char == ")":
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
            self.result = [[*x, *hold, *y] for y in subToken for x in self.result]
            hold.clear()
        elif suffix == "?":
            self._concat_hold()
            self.result.extend(tuple([*x, *y] for y in subToken for x in self.result))
        else:
            if self.optimizer is None:
                self.optimizer = Optimizer()
            substr = self.optimizer.compute(frozenset(subToken), omitOuterParen=True)
            hold.append(f"{substr}{suffix}" if self._is_char_block(substr) else f"({substr}){suffix}")

    def _concat_hold(self):
        hold = self.hold
        if hold:
            for s in self.result:
                s.extend(hold)
            hold.clear()

    def _reset_result(self):
        self.result.clear()
        self.result.append([])

    def _eat(self):
        """Consume one character in the token list."""
        try:
            char = self.token[self.index]
            self.index += 1
            return char
        except IndexError:
            return None

    def _peek(self):
        try:
            return self.token[self.index]
        except IndexError:
            return None

    def _eat_suffix(self):
        char = self._peek()
        if char not in self._suffixes:
            return

        suffixStart = self.index  # first char of suffix
        if char == "{":
            try:
                suffixEnd = self.token.index("}", suffixStart + 2)  # +2 so "{}" will not be matched
                assert self._rangeChars.issuperset(self.token[suffixStart + 1 : suffixEnd])
            except (ValueError, AssertionError):
                raise ValueError(f"Bad range format: {self.string}")
            self.index = suffixEnd + 1  # 1 char after "}"
            char = self._peek()

        while char in self._repetitions:
            self.index += 1
            char = self._peek()

        suffixEnd = self.index  # first char after suffix
        if suffixStart < suffixEnd:
            return self._join_slice(suffixStart, suffixEnd)

    def _join_slice(self, start: int, stop: int) -> str:
        return "".join(self.token[start:stop])

    @property
    def string(self):
        try:
            return self._string if isinstance(self._string, str) else "".join(self._string)
        except TypeError:
            pass


class Optimizer:
    def __init__(self) -> None:
        self._solver = pywraplp.Solver.CreateSolver("CBC")
        self._solverQue = deque()
        self._solverPool = {}
        self._prefix = defaultdict(set)
        self._suffix = defaultdict(set)
        self._remain = set()

    @lru_cache(maxsize=4096)
    def compute(self, tokenSet: frozenset, omitOuterParen: bool = False) -> str:
        """Compute an optimized regular expression for the given tokenset."""

        tokenSet = set(tokenSet)
        if () in tokenSet:
            quantifier = "?"
            tokenSet.remove(())
        else:
            quantifier = ""

        if any(map(self._is_word, tokenSet)):
            return self._wordStrategy(tokenSet, quantifier, omitOuterParen)

        try:
            return self._charsetStrategy(tokenSet, quantifier)
        except KeyError:
            return ""

    @staticmethod
    def _charsetStrategy(tokenSet: set, quantifier: str = "") -> str:

        if len(tokenSet) > 1:
            char = sorted(chain.from_iterable(tokenSet))

            if ("]",) in tokenSet:
                char.insert(0, char.pop(char.index("]")))
            if ("-",) in tokenSet:
                char.append(char.pop(char.index("-")))

            return f'[{"".join(char)}]{quantifier}'

        return f"{tokenSet.pop()[0]}{quantifier}"

    def _wordStrategy(self, tokenSet: set, quantifier: str, omitOuterParen: bool) -> str:

        tokenSetLength = len(tokenSet)
        if tokenSetLength == 1:
            string = "".join(*tokenSet)
            return f"({string}){quantifier}" if quantifier else string

        prefix = self._prefix
        suffix = self._suffix
        remain = self._remain
        groupKeys = self._solverPool.keys()
        result = []
        que = deque()
        segment = {}
        connection = {}
        connectionKeys = set()

        for token in tokenSet:
            left = {token[:i]: token[i:] for i in range(1, len(token))}
            right = {v: k for k, v in left.items()}
            left[token] = right[token] = ()
            segment[token] = left, right
            for i in left:
                prefix[i].add(token)
            for i in right:
                suffix[i].add(token)

        que.append((self._filter_affix(prefix), self._filter_affix(suffix)))
        prefix.clear()
        suffix.clear()

        if quantifier:
            tokenSet.add(())
            quantifier = ""
            tokenSetLength += 1

        while que:

            prefix, suffix = que.popleft()
            if not (prefix or suffix):
                continue

            for key, i, j in self._process_group(prefix, suffix, segment, tokenSet):
                connectionKeys.add(key)
                if key not in connection:
                    string = f"{self.compute(frozenset(i))}{self.compute(frozenset(j))}"
                    length = len(key)
                    value = (
                        sum(map(len, chain.from_iterable(key)))
                        + 0.999 * length
                        + 2 * (length == tokenSetLength)
                        - len(string)
                        - 1
                    )
                    connection[key] = (value, string) if value > 0 else None

            for optimal in self._optimize_group(connectionKeys, connection):

                remain.update(*groupKeys)
                for key in optimal:
                    result.append(connection[key][1])
                    remain.difference_update(key)
                    tokenSet.difference_update(key)

                if remain:
                    target = self._copy_affix if connectionKeys else self._update_affix
                    que.append((target(prefix, remain), target(suffix, remain)))
                    remain.clear()

        if () in tokenSet:
            quantifier = "?"
            tokenSet.remove(())

        if tokenSet:
            chars = set(filterfalse(self._is_word, tokenSet))
            if chars:
                tokenSet.difference_update(chars)
                result.append(self._charsetStrategy(chars))
            result.extend(map("".join, tokenSet))

        result.sort()
        string = "|".join(result)

        if quantifier or len(result) > 1 and not omitOuterParen:
            return f"({string}){quantifier}"
        return string

    @staticmethod
    @lru_cache(maxsize=512)
    def _is_word(token: tuple):
        return sum(map(len, token)) > 1

    @staticmethod
    def _filter_affix(d: dict) -> dict:
        """Keep groups which divide the same words at max common subsequence, and remove single member groups.

        - Example: (AB: ABC, ABD), (A: ABC, ABD), (ABC: ABC): only the first item will be keeped.
        """
        tmp = {}
        for k, v in d.items():
            if len(v) > 1:
                key = frozenset(v)
                length = len(k)
                if tmp.get(key, (0,))[0] < length:
                    tmp[key] = length, k
        return {v[1]: d[v[1]] for v in tmp.values()}

    @staticmethod
    def _update_affix(d: dict, r: set) -> dict:
        for v in d.values():
            v.intersection_update(r)
        return {k: v for k, v in d.items() if len(v) > 1}

    @staticmethod
    def _copy_affix(d: dict, r: set) -> dict:
        return {k: i for k, v in d.items() if len(i := v.intersection(r)) > 1}

    @staticmethod
    def _process_group(prefix: dict, suffix: dict, segment: dict, tokenSet: set):
        mirror = defaultdict(list)
        all_in_set = tokenSet.issuperset
        frozen_union = frozenset().union

        for k, v in prefix.items():
            mirror[frozenset(segment[j][0][k] for j in v)].append(k)
        for k, v in mirror.items():
            if all_in_set(k):
                v.append(())
                yield frozenset(x + y for x in v for y in k), v, k
            else:
                yield frozen_union(*map(prefix.get, v)), v, k

        mirror.clear()
        for k, v in suffix.items():
            mirror[frozenset(segment[j][1][k] for j in v)].append(k)
        for k, v in mirror.items():
            if all_in_set(k):
                v.append(())
                yield frozenset(x + y for x in k for y in v), k, v
            else:
                yield frozen_union(*map(suffix.get, v)), k, v

    def _optimize_group(self, unvisited: set, connection: dict):
        """Groups combinations in the way that each group is internally connected with common
        members. Then for each group, find the best non-overlapping members to reach the maximum
        length reduction.

        - Yield: Optimal keys
        - The overall group of each result should be read via self._solverPool.keys()
        - The input set (1st arg) will be finally emptied, which is an indication of the last group.
        """

        if len(unvisited) == 1:
            key = unvisited.pop()
            if connection[key]:
                yield (key,)
            return

        solver = self._solver
        objective = solver.Objective()
        que = self._solverQue
        pool = self._solverPool

        while unvisited:

            index = 0
            currentKey = next(iter(unvisited))
            value = connection[currentKey]
            if value:
                currentVar = solver.BoolVar(f"{index}")
                objective.SetCoefficient(currentVar, value[0])
                index += 1
            else:
                currentVar = None
            pool[currentKey] = currentVar
            que.append(currentKey)

            while que:
                currentKey = que.popleft()
                currentVar = pool[currentKey]
                unvisited.remove(currentKey)
                if () in currentKey:
                    currentKey = currentKey.difference(((),))

                for nextKey in filterfalse(currentKey.isdisjoint, unvisited):
                    try:
                        nextVar = pool[nextKey]
                    except KeyError:
                        value = connection[nextKey]
                        if value:
                            nextVar = solver.BoolVar(f"{index}")
                            objective.SetCoefficient(nextVar, value[0])
                            index += 1
                        else:
                            nextVar = None
                        pool[nextKey] = nextVar
                        que.append(nextKey)

                    if currentVar and nextVar:
                        solver.Add(currentVar + nextVar <= 1)

            if index > 0:

                if index == 1:
                    yield (next(filter(pool.get, pool)),)
                else:
                    objective.SetMaximization()
                    if solver.Solve() != solver.OPTIMAL:
                        raise RuntimeError("MIP Solver failed.")
                    yield (k for k, v in pool.items() if v and v.solution_value() == 1)

                solver.Clear()

            pool.clear()


class Regen:
    def __init__(self, wordlist: list) -> None:
        if not isinstance(wordlist, (list, tuple, set)):
            raise TypeError("Input should be a list of strings.")

        parser = Parser()
        self._tokens = frozenset(chain.from_iterable(map(parser.parse, wordlist)))
        self.wordlist = wordlist
        self._text = None
        self._regex = [None, None]

    def __repr__(self):
        return f"{self.__class__.__name__}({self.wordlist})"

    def to_text(self):
        """Extract the regular expressions to a list of corresponding words."""
        if self._text is None:
            self._text = sorted(map("".join, self._tokens))
        return iter(self._text)

    def to_regex(self, omitOuterParen: bool = False) -> str:
        """Return an optimized regular expression matching all the words.

        :omitOuterParen: If True, the outmost parentheses (if any) will be omited.
        """
        if not isinstance(omitOuterParen, bool):
            raise TypeError("omitOuterParen should be bool.")
        if self._regex[omitOuterParen] is None:
            self._regex[omitOuterParen] = Optimizer().compute(self._tokens, omitOuterParen=omitOuterParen)
        return self._regex[omitOuterParen]

    def verify_result(self):
        try:
            regex = next(i for i in self._regex if i is not None)
        except StopIteration:
            regex = self.to_regex()

        if self._tokens != Regen([regex])._tokens:
            raise ValueError("Extraction from computed regex is different from that of original wordlist.")

        pattern = re.compile(regex)
        for i in filterfalse(pattern.fullmatch, frozenset(chain(self.wordlist, self.to_text()))):
            if _specials.isdisjoint(i):
                raise ValueError(f"Computed regex does not fully match this word: '{i}'")

        return True


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
        wordlist = args.word
    else:
        wordlist = tuple(filter(None, args.file.read().splitlines()))
        args.file.close()

    regen = Regen(wordlist)

    if args.mode == "extract":
        for word in regen.to_text():
            print(word)

    else:
        regex = regen.to_regex(omitOuterParen=args.omit)
        print(regex)

        if args.verify:
            print("\nLength:", len(regex))
            print("Verifying... ", end="")
            try:
                regen.verify_result()
            except ValueError as e:
                print("failed:", e)
            else:
                print("passed.")


if __name__ == "__main__":
    main()
