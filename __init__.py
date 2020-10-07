import re
from collections import defaultdict, deque
from functools import lru_cache
from itertools import chain, filterfalse
from typing import FrozenSet, Iterable, List, Set, Tuple

from ortools.linear_solver.pywraplp import Solver


class Tokenizer:
    _rangeChars = frozenset(r"0123456789,")
    _psplit = re.compile(r"[^\\]|\\.")

    def __init__(self, string: Iterable) -> None:
        try:
            self.tokens = self._psplit.findall(string)
        except TypeError:
            if not isinstance(string, (list, tuple)):
                raise TypeError(f"Wrong type feed to tokenizer: {string}")
            self.tokens = string

        self.string = string
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
        suffixStart = self.index  # first char of suffix
        char = self.peek()
        if char == "{":
            try:
                suffixEnd = self.tokens.index("}", suffixStart + 2)  # +2 so "{}" will not be matched
                assert self._rangeChars.issuperset(self.tokens[suffixStart + 1 : suffixEnd])
            except (ValueError, AssertionError):
                raise ValueError(f"Bad range format: {self.string}")
            self.index = suffixEnd + 1  # 1 char after "}"
            char = self.peek()
        try:
            while char in "*?+":
                self.confirm()
                char = self.peek()
        except TypeError:  # Reach the end, char == None
            pass
        suffixEnd = self.index  # first char after suffix
        if suffixStart < suffixEnd:
            return self.get_substr(suffixStart, suffixEnd)

    def get_substr(self, start=0, stop=None):
        return "".join(self.tokens[start:stop])


class Extractor:

    _specials = frozenset("{}()[]|?*+")
    _repetitions = frozenset("*?+{")

    def __init__(self, string) -> None:
        self._string = string
        self._text = None
        if not string:
            self.result = ()
            return

        token = Tokenizer(string)
        if self._specials.isdisjoint(token.tokens):  # Not regex we can handle
            self.result = (tuple(token.tokens),)
            self._text = (self._get_string(),)
            return

        repetitions = self._repetitions
        result = []
        subresult = [[]]
        hold = []
        charset = []

        char = token.eat()
        while char:

            if char == "|":
                self._transfer(hold, subresult, result)
                char = token.eat()
                continue

            if char == "[":
                start = token.index - 1  # including "["
                while True:
                    char = token.eat()
                    if not char:
                        raise ValueError(f"Bad charactor set: {self._get_string()}")
                    if char == "[":
                        raise ValueError(f"Nested character set: {self._get_string()}")
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
                    self._concat(hold, subresult)
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
                    raise ValueError(f"Unbalanced parenthesis: {self._get_string()}")

                end = token.index - 1  # index of ")"
                substr = Extractor(token.tokens[start:end])

                suffix = token.eat_suffix()
                if not suffix:
                    subresult = [[*x, *hold, *y] for x in subresult for y in substr.result]
                    hold.clear()
                elif suffix == "?":
                    self._concat(hold, subresult)
                    subresult.extend(tuple([*x, *y] for x in subresult for y in substr.result))
                else:
                    char = f"{Optimizer(substr).result}{suffix}"
                    hold.append(char)

            elif char in repetitions:
                raise ValueError(f"Invalid charactor '{char}': {self._get_string()}")

            else:
                suffix = token.eat_suffix()
                if not suffix:
                    hold.append(char)
                elif suffix == "?":
                    self._concat(hold, subresult)
                    subresult.extend(tuple([*x, char] for x in subresult))
                else:
                    hold.append(f"{char}{suffix}")

            char = token.confirm()

        self._transfer(hold, subresult, result)
        self.result = tuple(map(tuple, result))

    @staticmethod
    def _concat(hold: List, subresult: List[List]):
        if hold:
            for s in subresult:
                s.extend(hold)
            hold.clear()

    @classmethod
    def _transfer(cls, hold: List, subresult: List[List], result: List[List]):
        """Concat hold with subresult, then transfer subresult to self.result."""
        cls._concat(hold, subresult)
        result.extend(subresult)
        subresult.clear()
        subresult.append([])

    def _get_string(self):
        if not isinstance(self._string, str):
            self._string = "".join(self._string)
        return self._string

    def get_text(self):
        if self._text is None:
            self._text = tuple(map("".join, self.result))
        return self._text


class Optimizer:

    _charSetFront = tuple((c,) for c in "]^")
    _charSetEnd = tuple((c,) for c in "-")

    def __init__(self, *extracted: Extractor) -> None:
        self._connection = {}
        self.solver = Solver.CreateSolver("RegexOptimizer", "CBC")
        self.result = self._compute_regex(frozenset(t for e in extracted for t in e.result))
        del self._connection

    @lru_cache(maxsize=4096)
    def _compute_regex(self, tokenSet: FrozenSet[Tuple[str]]) -> str:

        tokenSet = set(tokenSet)

        if () in tokenSet:
            qmark = "?"
            tokenSet.remove(())
        else:
            qmark = ""

        if not tokenSet:
            return ""

        if any(len(w) > 1 for w in tokenSet):
            if len(tokenSet) == 1:
                string = "".join(tokenSet.pop())
                return f"({string})?" if qmark else string

            result = []
            que = deque()
            lgroup = defaultdict(set)
            rgroup = defaultdict(set)
            lgroupReverse = defaultdict(list)
            rgroupReverse = defaultdict(list)
            segment = {}
            connection = self._connection
            connectionKeys = set()
            subResult = []
            subTokenSet = set()

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

                for group, target, i in (lgroup, lgroupReverse, 0), (rgroup, rgroupReverse, 1):
                    for k, v in group.items():
                        target[frozenset(segment[j][i][k] for j in v)].append(k)

                left = ((frozenset(j for i in v for j in lgroup[i]), k, v) for k, v in lgroupReverse.items())
                right = ((frozenset(j for i in v for j in rgroup[i]), v, k) for k, v in rgroupReverse.items())

                for target in (left, right):
                    for key, i, j in target:
                        connectionKeys.add(key)
                        if key not in connection:
                            # [0]: prefix
                            # [1]: suffix
                            # [2]: concatLength + 0.1, then reduced length: concatLength - stringLength
                            # [3]: computed regex string
                            connection[key] = [i, j, sum(map(len, chain(*key))) + len(key) - 0.9, None]

                lgroupReverse.clear()
                rgroupReverse.clear()

                for group in self._group_keys(connectionKeys):

                    for key in group:
                        v = connection[key]

                        if v[3] is None:
                            v[3] = f"{self._compute_regex(frozenset(v[0]))}{self._compute_regex(frozenset(v[1]))}"
                            v[2] -= len(v[3])

                        if v[2] > 0:
                            subResult.append(key)

                    if not subResult:
                        continue

                    subTokenSet.update(chain(*group))

                    for key in self._mip_solver(subResult):
                        result.append(connection[key][3])
                        subTokenSet.difference_update(key)
                        tokenSet.difference_update(key)

                    if subTokenSet:
                        subLgroup = {k: i for k, v in lgroup.items() if len(i := v.intersection(subTokenSet)) > 1}
                        subRgroup = {k: i for k, v in rgroup.items() if len(i := v.intersection(subTokenSet)) > 1}
                        que.append((subLgroup, subRgroup))
                        subTokenSet.clear()

            if tokenSet:
                chars = frozenset(i for i in tokenSet if len(i) == 1)
                if chars:
                    result.append(self._compute_regex(chars))
                    tokenSet.difference_update(chars)
                result.extend(map("".join, tokenSet))

            result.sort()
            string = "|".join(result)

            if qmark or len(result) > 1:
                string = f"({string}){qmark}"

            return string

        elif len(tokenSet) > 1:
            char = sorted(chain(*tokenSet))

            for c in tokenSet.intersection(self._charSetFront):
                char.insert(0, char.pop(char.index(c[0])))
            for c in tokenSet.intersection(self._charSetEnd):
                char.append(char.pop(char.index(c[0])))

            return f'[{"".join(char)}]{qmark}'

        return f"{tokenSet.pop()[0]}{qmark}"

    @staticmethod
    def _filter_group(d: dict):
        """Keep groups which divide the same words at max common subsequence, and remove single member groups.

        Example: (AB: ABC, ABD), (A: ABC, ABD), (ABC: ABC)
        only the first item will be keeped.
        """
        tmp = {}
        for k, v in d.items():
            if len(v) <= 1:
                continue
            key = frozenset(v)
            length = len(k)
            try:
                if tmp[key][0] >= length:
                    continue
            except KeyError:
                pass
            tmp[key] = length, k

        return {v[1]: d[v[1]] for v in tmp.values()}

    @staticmethod
    def _group_keys(connectionKeys: Set[FrozenSet]):
        """Group keys with common members together.

        The input set will be emptied."""
        que = deque()
        while connectionKeys:
            currentVert = connectionKeys.pop()
            que.append(currentVert)
            currentGroup = [currentVert]
            while que:
                currentVert = que.popleft()
                connected = tuple(filterfalse(currentVert.isdisjoint, connectionKeys))
                connectionKeys.difference_update(connected)
                currentGroup.extend(connected)
                if not connectionKeys:
                    break
                que.extend(connected)
            yield currentGroup

    def _mip_solver(self, subResult: List[FrozenSet]):
        """Find the non-overlapping group with the maximum sum of length reduction.

        The input list will be emptied."""

        if len(subResult) == 1:
            yield subResult.pop()
            return

        solver = self.solver
        connection = self._connection
        pool = {k: solver.BoolVar(str(i)) for i, k in enumerate(subResult)}
        objective = solver.Objective()
        objective.SetMaximization()

        while subResult:
            currentkey = subResult.pop()
            currentVar = pool[currentkey]
            objective.SetCoefficient(currentVar, connection[currentkey][2])
            for k in filterfalse(currentkey.isdisjoint, subResult):
                solver.Add(currentVar + pool[k] <= 1)

        if solver.Solve() != solver.OPTIMAL:
            raise RuntimeError("MIP Solver failed.")

        for k, v in pool.items():
            if v.solution_value() == 1:
                yield k

        solver.Clear()


def test_regex(regex: str, wordlist: list):
    extracted = Extractor(regex).get_text()
    assert sorted(wordlist) == sorted(extracted), "Extracted regex is different from original words."

    regex = re.compile(regex)
    for i in filterfalse(regex.fullmatch, wordlist):
        assert re.search(r"[*+{}]", i), f"Regex matching test failed: '{i}'"
