import re
from collections import defaultdict, deque
from functools import lru_cache
from itertools import chain, filterfalse

from ortools.linear_solver import pywraplp

_specials = frozenset("{}()[]|?*+")


class Tokenizer:

    _rangeChars = frozenset("0123456789,")
    _repetitions = frozenset("*?+{")
    _suffixes = frozenset("*?+")
    _psplit = re.compile(r"[^\\]|\\.").findall

    def __init__(self, string: str) -> None:

        self._string = string

        try:
            self.tokens = self._psplit(string)
        except TypeError:
            if not isinstance(string, (list, tuple)):
                raise TypeError(f"Wrong type feed to tokenizer: {string}")
            self.tokens = string

        self.index = 0

    def eat(self):
        try:
            char = self.tokens[self.index]
        except IndexError:
            char = None
        else:
            self.index += 1
        return char

    def peek(self):
        index = self.index
        try:
            char = self.tokens[index]
        except IndexError:
            char = None
        self.peekindex = index + 1
        self.peekchar = char
        return char

    def confirm(self):
        """Confirm (eat) the last peek."""
        try:
            self.index = self.peekindex
            return self.peekchar
        except AttributeError as e:
            raise RuntimeError(f"Confirm before peek: {self.string}")

    def eat_suffix(self):
        char = self.peek()
        if char not in self._repetitions:
            return

        suffixStart = self.index  # first char of suffix
        if char == "{":
            try:
                suffixEnd = self.tokens.index("}", suffixStart + 2)  # +2 so "{}" will not be matched
                assert self._rangeChars.issuperset(self.tokens[suffixStart + 1 : suffixEnd])
            except (ValueError, AssertionError):
                raise ValueError(f"Bad range format: {self.string}")
            self.index = suffixEnd + 1  # 1 char after "}"
            char = self.peek()

        try:
            while char in self._suffixes:
                self.confirm()
                char = self.peek()
        except TypeError:  # Reach the end, char == None
            pass

        suffixEnd = self.index  # first char after suffix
        if suffixStart < suffixEnd:
            return self.get_substr(suffixStart, suffixEnd)

    def get_substr(self, start=0, stop=None) -> str:
        return "".join(self.tokens[start:stop])

    @property
    def string(self):
        return self._string if isinstance(self._string, str) else "".join(self._string)


class Parser:
    @classmethod
    def parse(cls, string: str) -> frozenset:
        """Convert a regular expression to a tokenset."""

        if not string:
            return frozenset()

        token = Tokenizer(string)
        if _specials.isdisjoint(token.tokens):  # Not regex we can handle
            return frozenset((tuple(token.tokens),))

        result = []
        subresult = [[]]
        hold = []
        charset = []

        char = token.eat()
        while char:

            if char == "|":
                cls._transfer(hold, subresult, result)
                char = token.eat()
                continue

            if char == "[":
                start = token.index - 1  # including "["
                while True:
                    char = token.eat()
                    if not char:
                        raise ValueError(f"Bad character set: {token.string}")
                    if char == "[":
                        raise ValueError(f"Nested character set: {token.string}")
                    if char == "]" and charset:
                        break
                    if char == "-" and token.peek() != "]":
                        try:
                            lo = charset.pop()
                            hi = token.eat()
                            char = f"[{lo}{char}{hi}]"
                        except IndexError:  # "-" is the first char in chaset
                            pass
                    charset.append(char)

                suffix = token.eat_suffix()
                if not suffix:
                    subresult = [[*x, *hold, y] for x in subresult for y in charset]
                    hold.clear()
                elif suffix == "?":
                    cls._concat(hold, subresult)
                    subresult.extend(tuple([*x, y] for x in subresult for y in charset))
                else:
                    end = token.index  # from "[" to the end of suffix
                    char = token.get_substr(start, end)
                    hold.append(char)

                charset.clear()

            elif char == "(":
                balance = 0
                start = token.index  # 1 char after "("
                while char:
                    if char == "(":
                        balance += 1
                    elif char == ")":
                        balance -= 1
                        if balance == 0:
                            break
                    char = token.eat()
                else:
                    raise ValueError(f"Unbalanced parenthesis: {token.string}")

                end = token.index - 1  # index of ")"
                subToken = cls.parse(token.tokens[start:end])

                suffix = token.eat_suffix()
                if not suffix:
                    subresult = [[*x, *hold, *y] for x in subresult for y in subToken]
                    hold.clear()
                elif suffix == "?":
                    cls._concat(hold, subresult)
                    subresult.extend(tuple([*x, *y] for x in subresult for y in subToken))
                else:
                    char = f"{Optimizer().compute(subToken)}{suffix}"
                    hold.append(char)

            elif char in _specials:
                raise ValueError(f"Invalid character '{char}': '{token.string}'")

            else:
                suffix = token.eat_suffix()
                if not suffix:
                    hold.append(char)
                elif suffix == "?":
                    cls._concat(hold, subresult)
                    subresult.extend(tuple([*x, char] for x in subresult))
                else:
                    hold.append(f"{char}{suffix}")

            char = token.confirm()

        cls._transfer(hold, subresult, result)
        return frozenset(map(tuple, result))

    @staticmethod
    def _concat(hold: list, subresult: list):
        if hold:
            for s in subresult:
                s.extend(hold)
            hold.clear()

    @classmethod
    def _transfer(cls, hold: list, subresult: list, result: list):
        """Concat hold with subresult, then transfer subresult to self.result."""
        cls._concat(hold, subresult)
        result.extend(subresult)
        subresult.clear()
        subresult.append([])


class Optimizer:

    _charsetFront = tuple((c,) for c in "]^")
    _charsetEnd = frozenset((c,) for c in "-")

    def __init__(self) -> None:
        self._solver = pywraplp.Solver.CreateSolver("CBC")

    @lru_cache(maxsize=4096)
    def compute(self, tokenSet: frozenset) -> str:
        """Compute an optimal regular expression for the given tokenset."""

        tokenSet = set(tokenSet)
        if () in tokenSet:
            qmark = "?"
            tokenSet.remove(())
        else:
            qmark = ""

        if any(len(i) > 1 for i in tokenSet):
            return self._wordStrategy(tokenSet, qmark)

        try:
            return self._charsetStrategy(tokenSet, qmark)
        except KeyError:
            return ""

    def _charsetStrategy(self, tokenSet: set, qmark: str = "") -> str:

        if len(tokenSet) > 1:
            char = sorted(chain.from_iterable(tokenSet))

            for c in tokenSet.intersection(self._charsetFront):
                char.insert(0, char.pop(char.index(c[0])))
            for c in tokenSet.intersection(self._charsetEnd):
                char.append(char.pop(char.index(c[0])))

            return f'[{"".join(char)}]{qmark}'

        return f"{tokenSet.pop()[0]}{qmark}"

    def _wordStrategy(self, tokenSet: set, qmark: str) -> str:

        tokenSetLength = len(tokenSet)

        if tokenSetLength == 1:
            string = "".join(*tokenSet)
            return f"({string})?" if qmark else string

        result = []
        que = deque()
        lgroup = defaultdict(set)
        rgroup = defaultdict(set)
        lgroupMirror = defaultdict(list)
        rgroupMirror = defaultdict(list)
        segment = {}
        connection = {}
        connectionKeys = set()
        remain = set()

        for token in tokenSet:
            left = {token[i:]: token[:i] for i in range(1, len(token))}
            right = {v: k for k, v in left.items()}
            left[token] = right[token] = ()
            segment[token] = (left, right)
            for k in left:
                lgroup[k].add(token)
            for k in right:
                rgroup[k].add(token)

        lgroup = self._filter_group(lgroup)
        rgroup = self._filter_group(rgroup)
        que.append((lgroup, rgroup))

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
                        sum(map(len, chain.from_iterable(key)))
                        + 0.999 * length
                        + 2 * (length == tokenSetLength)
                        - len(string)
                        - 1
                    )
                    connection[key] = (value, string) if value > 0 else None
                connectionKeys.add(key)

            lgroupMirror.clear()
            rgroupMirror.clear()

            for group, optimal in self._group_optimize(connectionKeys, connection):

                remain.update(*group)
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
            chars = {i for i in tokenSet if len(i) == 1}
            if chars:
                tokenSet.difference_update(chars)
                result.append(self._charsetStrategy(chars))
            result.extend(map("".join, tokenSet))

        result.sort()
        string = "|".join(result)

        if len(result) > 1 or qmark:
            return f"({string}){qmark}"
        return string

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

        - Yields: (Group keys, Optimal keys)
        - The input set (unvisited) will be emptied.
        """

        if len(unvisited) == 1:
            key = unvisited.pop()
            if connection[key]:
                yield (), (key,)
            return

        solver = self._solver
        objective = solver.Objective()
        que = deque()
        pool = {}

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
                    optimal = (next(filter(pool.get, pool)),)
                else:
                    objective.SetMaximization()
                    if solver.Solve() != solver.OPTIMAL:
                        raise RuntimeError("MIP Solver failed.")
                    optimal = (k for k, v in pool.items() if v and v.solution_value() == 1)

                yield pool.keys(), optimal
                solver.Clear()

            pool.clear()


class Regen:
    def __init__(self, wordlist: list) -> None:

        if not isinstance(wordlist, (list, tuple, set)):
            raise TypeError("Input should be a list of strings.")

        self.wordlist = wordlist
        parser = Parser()
        self._tokens = frozenset.union(*map(parser.parse, wordlist))
        self._textList = self._regex = None

    def to_text(self):
        """Extract a regular expression to a list of corresponding plain text."""
        if self._textList is None:
            self._textList = sorted(map("".join, self._tokens))
        return self._textList

    def to_regex(self):
        """Output a single regular expression matching all the strings."""
        if self._regex is None:
            self._regex = Optimizer().compute(self._tokens)
        return self._regex

    def verify_result(self):
        text = self.to_text()
        regex = self.to_regex()

        secondText = Regen([regex]).to_text()
        assert text == secondText, "Extraction from computed regex is different from that of original wordlist."

        pattern = re.compile(regex)
        for i in filterfalse(pattern.fullmatch, chain(self.wordlist, text)):
            assert not _specials.isdisjoint(i), f"Computed regex does not full match this word: '{i}'"
