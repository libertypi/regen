import re
from collections import defaultdict, deque
from functools import lru_cache
from itertools import chain, filterfalse

from ortools.linear_solver import pywraplp


class Tokenizer:
    _rangeChars = frozenset(r"0123456789,")
    _psplit = re.compile(r"[^\\]|\\.")

    def __init__(self, string: str) -> None:
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

    def __init__(self, string: str) -> None:
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
                        raise ValueError(f"Bad character set: {self._get_string()}")
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
                raise ValueError(f"Invalid character '{char}': {self._get_string()}")

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
        self._solver = pywraplp.Solver.CreateSolver("RegexOptimizer", "CBC")
        self.result = self._compute_regex(frozenset(chain.from_iterable(e.result for e in extracted)))

    @lru_cache(maxsize=4096)
    def _compute_regex(self, tokenSet: frozenset) -> str:

        tokenSet = set(tokenSet)

        if () in tokenSet:
            qmark = "?"
            tokenSet.remove(())
        else:
            qmark = ""

        tokenSetLength = len(tokenSet)
        if not tokenSetLength:
            return ""

        if any(len(i) > 1 for i in tokenSet):

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
                        string = f"{self._compute_regex(frozenset(i))}{self._compute_regex(frozenset(j))}"
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
                        subLgroup = {k: i for k, v in lgroup.items() if len(i := v.intersection(remain)) > 1}
                        subRgroup = {k: i for k, v in rgroup.items() if len(i := v.intersection(remain)) > 1}
                        que.append((subLgroup, subRgroup))
                        remain.clear()

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

        elif tokenSetLength > 1:
            char = sorted(chain.from_iterable(tokenSet))

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
            if tmp.get(key, (0,))[0] < length:
                tmp[key] = length, k

        return {v[1]: d[v[1]] for v in tmp.values()}

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
                yield (key,), (key,)
            return

        solver = self._solver
        que = deque()
        pool = {}

        while unvisited:
            objective = solver.Objective()

            index = 0
            currentKey = next(iter(unvisited))
            value = connection[currentKey]
            if value:
                currentVar = solver.BoolVar(str(index))
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
                            nextVar = solver.BoolVar(str(index))
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
                    optimal = (k for k, v in pool.items() if v)
                else:
                    objective.SetMaximization()
                    if solver.Solve() != solver.OPTIMAL:
                        raise RuntimeError("MIP Solver failed.")
                    optimal = (k for k, v in pool.items() if v and v.solution_value() == 1)

                yield pool.keys(), optimal
                solver.Clear()

            pool.clear()


def test_regex(regex: str, wordlist: list):
    extracted = Extractor(regex).get_text()
    assert sorted(wordlist) == sorted(extracted), "Extracted regex is different from original words."

    pattern = re.compile(regex)
    for i in filterfalse(pattern.fullmatch, wordlist):
        assert re.search(r"[*+{}]", i), f"Regex matching test failed: '{i}'"
