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
"""


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
    _parenFinder = re.compile(r"(?<!\\)[()]").finditer

    def __init__(self) -> None:
        self.result = [[]]
        self.hold = []
        self.charset = []
        self.index = 0
        self.subParser = self.optimizer = None

    def parse(self, string):
        """Convert a regular expression to a tokenset."""

        if not string:
            return

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

        char = self._eat()
        while char:

            if char == "|":
                self._concat_hold()
                yield from map(tuple, self.result)
                self._reset_result()

            elif char == "[":
                self._charsetStrategy()

            elif char == "(":
                self._parenStrategy(char)

            elif char in _specials:
                raise ValueError(f"Invalid character '{char}': '{self.string}'")

            else:
                suffix = self._eat_suffix()
                if not suffix:
                    hold.append(char)
                elif suffix == "?":
                    self._concat_hold()
                    self.result.extend(tuple([*x, char] for x in self.result))
                else:
                    hold.append(f"{char}{suffix}")

            char = self._eat()

        self._concat_hold()
        yield from map(tuple, self.result)
        self._reset_result()
        self.index = 0
        self.token = self._string = None

    def _charsetStrategy(self):
        start = self.index - 1  # including "["
        hold = self.hold
        charset = self.charset

        char = self._eat()
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
                        hi = self._eat()
                        char = f"[{lo}{char}{hi}]"
                    except IndexError:  # "-" is the first char in charset
                        pass
                charset.append(char)
                char = self._eat()
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

    def _parenStrategy(self, char: str):

        start = self.index  # 1 char after "("
        hold = self.hold
        balance = 0

        while char:
            if char == "(":
                balance += 1
            elif char == ")":
                balance -= 1
                if balance == 0:
                    break
            char = self._eat()
        else:
            raise ValueError(f"Unbalanced parenthesis: {self.string}")

        end = self.index - 1  # index of ")"

        if self.subParser is None:
            self.subParser = Parser()

        subToken = self.subParser.parse(self.token[start:end])
        suffix = self._eat_suffix()

        if not suffix:
            self.result = [[*x, *hold, *y] for y in subToken for x in self.result]
            hold.clear()
        elif suffix == "?":
            self._concat_hold()
            self.result.extend(tuple([*x, *y] for y in subToken for x in self.result))
        else:
            if self.optimizer is None:
                self.optimizer = Optimizer()
            substr = self.optimizer.compute(frozenset(subToken))

            if self._is_block(substr):
                hold.append(f"{substr}{suffix}")
            else:
                hold.append(f"({substr}){suffix}")

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

    @classmethod
    def _is_block(cls, string: str):
        """Return True if the string is like: A, [AB], (AB|CD)."""
        if re.fullmatch(r"\\?.|\[(\^?\])?([^]]|\\\])*\]", string):
            return True
        return cls.is_parenthesized(string)

    @classmethod
    def is_parenthesized(cls, string: str):
        if string[0] != "(" or string[-1] != ")" or string[-2] == "\\":
            return False
        b = 0
        for c in cls._parenFinder(string, 1, len(string) - 1):
            if c.group() == "(":
                b += 1
            else:
                b -= 1
                if b < 0:
                    return False
        if b != 0:
            raise RuntimeError(f'Unbalanced brackets in regex: "{string}"')
        return True


class Optimizer:
    def __init__(self) -> None:
        self._solver = pywraplp.Solver.CreateSolver("CBC")
        self._solverQue = deque()
        self._solverPool = {}
        self._lgroup = defaultdict(set)
        self._rgroup = defaultdict(set)
        self._remain = set()

    @lru_cache(maxsize=4096)
    def compute(self, tokenSet: frozenset) -> str:
        """Compute an optimized regular expression for the given tokenset."""

        tokenSet = set(tokenSet)
        if () in tokenSet:
            quantifier = "?"
            tokenSet.remove(())
        else:
            quantifier = ""

        if any(self._token_length(i) > 1 for i in tokenSet):
            return self._wordStrategy(tokenSet, quantifier)

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

    def _wordStrategy(self, tokenSet: set, quantifier: str) -> str:

        tokenSetLength = len(tokenSet)

        if tokenSetLength == 1:
            string = "".join(*tokenSet)
            return f"({string}){quantifier}" if quantifier else string

        lgroup = self._lgroup
        rgroup = self._rgroup
        remain = self._remain
        groupKeys = self._solverPool.keys()
        result = []
        que = deque()
        lgroupMirror = defaultdict(list)
        rgroupMirror = defaultdict(list)
        segment = {}
        connection = {}
        connectionKeys = set()

        for token in tokenSet:
            left = {token[i:]: token[:i] for i in range(1, len(token))}
            right = {v: k for k, v in left.items()}
            left[token] = right[token] = ()
            segment[token] = (left, right)
            for k in left:
                lgroup[k].add(token)
            for k in right:
                rgroup[k].add(token)

        que.append((self._filter_group(lgroup), self._filter_group(rgroup)))
        lgroup.clear()
        rgroup.clear()

        while que:

            lgroup, rgroup = que.popleft()
            if not (lgroup or rgroup):
                continue

            for group, target, i in (lgroup, lgroupMirror, 0), (rgroup, rgroupMirror, 1):
                for k, v in group.items():
                    target[frozenset(segment[j][i][k] for j in v)].append(k)

            left = ((map(lgroup.get, v), k, v) for k, v in lgroupMirror.items())
            right = ((map(rgroup.get, v), v, k) for k, v in rgroupMirror.items())

            for key, i, j in chain(left, right):
                key = frozenset().union(*key)
                if key not in connection:
                    string = f"{self.compute(frozenset(i))}{self.compute(frozenset(j))}"
                    length = len(key)
                    value = (
                        sum(map(self._token_length, key))
                        + 0.999 * length
                        + 2 * (length == tokenSetLength)
                        - len(string)
                        - 1
                    )
                    connection[key] = (value, string) if value > 0 else None
                connectionKeys.add(key)

            lgroupMirror.clear()
            rgroupMirror.clear()

            for optimal in self._group_optimize(connectionKeys, connection):

                remain.update(*groupKeys)
                for key in optimal:
                    result.append(connection[key][1])
                    remain.difference_update(key)
                    tokenSet.difference_update(key)

                if remain:
                    target = self._copy_group if connectionKeys else self._update_group
                    subLgroup = target(lgroup, remain)
                    subRgroup = target(rgroup, remain)
                    que.append((subLgroup, subRgroup))
                    remain.clear()

        if tokenSet:
            chars = {i for i in tokenSet if self._token_length(i) == 1}
            if chars:
                tokenSet.difference_update(chars)
                result.append(self._charsetStrategy(chars))
            result.extend(map("".join, tokenSet))

        result.sort()
        string = "|".join(result)

        return f"({string}){quantifier}" if quantifier or len(result) > 1 else string

    @staticmethod
    @lru_cache(maxsize=512)
    def _token_length(token: tuple):
        return sum(map(len, token))

    @staticmethod
    def _filter_group(d: dict) -> dict:
        """Keep groups which divide the same words at max common subsequence, and remove single member groups.

        - Example: (AB: ABC, ABD), (A: ABC, ABD), (ABC: ABC): only the first item will be keeped.
        """
        tmp = {}
        for k, v in d.items():
            if len(v) <= 1:
                continue
            key = frozenset(v)
            length = len(k)
            if tmp.get(key, (0,))[0] < length:
                tmp[key] = length, k

        return {v[1]: d[v[1]] for v in tmp.values()}

    @staticmethod
    def _update_group(group: dict, refer: set) -> dict:
        for v in group.values():
            v.intersection_update(refer)
        return {k: v for k, v in group.items() if len(v) > 1}

    @staticmethod
    def _copy_group(group: dict, refer: set) -> dict:
        return {k: i for k, v in group.items() if len(i := v.intersection(refer)) > 1}

    def _group_optimize(self, unvisited: set, connection: dict):
        """Groups combinations in the way that each group is internally connected with common
        members. Then for each group, find the best non-overlapping members to reach the maximum
        length reduction.

        - Yield: Optimal keys
        - The overall group for each result should be read through self._solverPool.keys()
        - The input set (unvisited) will be finally emptied, which is an indication of the last group.
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

        self.wordlist = wordlist
        parser = Parser()
        self._tokens = frozenset(chain.from_iterable(map(parser.parse, wordlist)))
        self._textList = self._regex = None

    def to_text(self):
        """Extract the regular expressions to a list of corresponding words."""
        if self._textList is None:
            self._textList = sorted(map("".join, self._tokens))
        return self._textList

    def to_regex(self, omitOuterParen: bool = False):
        """Return an optimized regular expression matching all the words.

        :omitOuterParen: If True, the outmost parentheses (if any) will be omited.
        """
        if self._regex is None:
            self._regex = Optimizer().compute(self._tokens)

        if omitOuterParen and Parser.is_parenthesized(self._regex):
            return self._regex[1:-1]

        return self._regex

    def verify_result(self):
        text = self.to_text()
        regex = self.to_regex()

        secondText = Regen([regex]).to_text()
        assert text == secondText, "Extraction from computed regex is different from that of original wordlist."

        pattern = re.compile(regex)
        for i in filterfalse(pattern.fullmatch, frozenset(chain(self.wordlist, text))):
            assert not _specials.isdisjoint(i), f"Computed regex does not fully match this word: '{i}'"
