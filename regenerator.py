#!/usr/bin/env python3

"""
Main module for computing regular expressions from a list of strings/regex.

The libary expand a list of regex to a finite set of words, then generate a new
regular expression using linear optimization to find the near-shortest
combination. The computed regex should match exactly the same words as input.

The Regen class is intended for external uses. Other classes in the libary are
for internal uses only.

### Example:

    >>> from regenerator import Regen

    >>> wordlist = ['ABC', 'ABD', 'BBC', 'BBD']
    >>> regen = Regen(wordlist)
    >>> regen.to_regex()
    ... '[AB]B[CD]'

    >>> wordlist = ['[AB]B[CD]', 'XYZ']
    >>> regen = Regen(wordlist)

    >>> regen.to_text()
    ... ['ABC', 'ABD', 'BBC', 'BBD', 'XYZ']

    >>> regen.to_regex()
    ... '(XYZ|[AB]B[CD])'
    
    >>> regen.to_regex(omitOuterParen=True)
    ... 'XYZ|[AB]B[CD]'

### Author: `David Pi`
"""

__all__ = ["Regen"]

import re
from collections import defaultdict
from functools import lru_cache
from itertools import chain, compress, filterfalse

from ortools.sat.python import cp_model

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
                    hold.append(char + suffix)

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
            hold.append(substr + suffix if self._is_char_block(substr) else f"({substr}){suffix}")

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
        self._prefix = defaultdict(set)
        self._suffix = defaultdict(set)

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

        result = []
        segment = {}
        candidate = {}
        unvisitCand = set()
        mirror = defaultdict(list)
        frozenUnion = frozenset().union
        prefix = self._prefix
        suffix = self._suffix

        for token in tokenSet:
            left = {token[:i]: token[i:] for i in range(1, len(token))}
            right = {v: k for k, v in left.items()}
            left[token] = right[token] = ()
            segment[token] = left, right
            for k in left:
                prefix[k].add(token)
            for k in right:
                suffix[k].add(token)

        prefix = self._filter_affix(prefix)
        suffix = self._filter_affix(suffix)
        self._prefix.clear()
        self._suffix.clear()

        if quantifier:
            tokenSet.add(())
            tokenSetLength += 1
        factor = tokenSetLength + 1

        while prefix or suffix:

            for source, i in (prefix, 0), (suffix, 1):

                for k, v in source.items():
                    mirror[frozenset(segment[j][i][k] for j in v)].append(k)

                for k, v in mirror.items():
                    if tokenSet.issuperset(k):
                        key = k.union(*map(source.get, v))
                        v.append(())
                    else:
                        key = frozenUnion(*map(source.get, v))

                    if key not in candidate:
                        left = self.compute(k)
                        right = self.compute(frozenset(v))
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
            for key in self._optimize_group(unvisitCand, candidate):
                result.append(candidate[key][1])
                tokenSet.difference_update(key)

            if not tokenSet:
                break
            prefix = self._update_affix(prefix, tokenSet)
            suffix = self._update_affix(suffix, tokenSet)

        if quantifier:
            try:
                tokenSet.remove(())
            except KeyError:
                quantifier = ""

        if tokenSet:
            chars = set(filterfalse(self._is_word, tokenSet))
            if len(chars) > 2:
                tokenSet.difference_update(chars)
                result.append(self._charsetStrategy(chars))
            result.extend(map("".join, tokenSet))

        result.sort()
        string = "|".join(result)

        if len(result) > 1 and not omitOuterParen or quantifier:
            return f"({string}){quantifier}"
        return string

    @staticmethod
    @lru_cache(maxsize=512)
    def _is_word(token: tuple):
        return sum(map(len, token)) > 1

    @staticmethod
    def _filter_affix(d: dict) -> dict:
        """Keep groups which divide the same words at max common subsequence,
        and remove single member groups.

        - Example: (AB: ABC, ABD), (A: ABC, ABD), (ABC: ABC): only the first
          item will be keeped.
        """
        tmp = {}
        for k, v in d.items():
            if len(v) > 1:
                key = frozenset(v)
                val = len(k), k
                if tmp.setdefault(key, val) < val:
                    tmp[key] = val
        return {v: d[v] for _, v in tmp.values()}

    @staticmethod
    def _update_affix(d: dict, r: set) -> dict:
        for v in d.values():
            v.intersection_update(r)
        return {k: v for k, v in d.items() if len(v) > 1}

    @staticmethod
    def _optimize_group(unvisited: set, candidate: dict):
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
        while unvisited:

            currentKey = unvisited.pop()
            if all(map(currentKey.isdisjoint, unvisited)):
                yield currentKey
                continue

            model = cp_model.CpModel()
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
                    try:
                        nextVar = pool[nextKey]
                    except KeyError:
                        nextVar = model.NewBoolVar(f"{index}")
                        index += 1
                        pool[nextKey] = nextVar
                        stack.append(nextKey)
                    model.AddImplication(nextVar, currentVarNot)

            model.Maximize(
                cp_model.LinearExpr.ScalProd(
                    pool.values(),
                    tuple(candidate[k][0] for k in pool),
                )
            )

            solver = cp_model.CpSolver()
            status = solver.Solve(model)
            if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                raise RuntimeError(f"CP-SAT Solver failed, status: {solver.StatusName(status)}")

            yield from compress(pool, map(solver.BooleanValue, pool.values()))
            pool.clear()


class Regen:
    def __init__(self, wordlist: list) -> None:
        if not isinstance(wordlist, (list, tuple, set)):
            raise TypeError("Input should be a list of strings.")

        parser = Parser()
        self._tokens = frozenset(chain.from_iterable(map(parser.parse, wordlist)))
        self.wordlist = wordlist
        self._text = self.optimizer = None
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
            raise TypeError("omitOuterParen should be a bool.")

        if self._regex[omitOuterParen] is None:
            if not self.optimizer:
                self.optimizer = Optimizer()
            self._regex[omitOuterParen] = self.optimizer.compute(self._tokens, omitOuterParen=omitOuterParen)

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
