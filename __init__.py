import re
from collections import defaultdict, deque
from functools import lru_cache
from typing import Iterable, List, Tuple


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
    def _concat(hold: list, subresult: List[list]):
        if hold:
            for s in subresult:
                s.extend(hold)
            hold.clear()

    @classmethod
    def _transfer(cls, hold: list, subresult: List[list], result: List[list]):
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
    def __init__(self, *extracted: Extractor) -> None:
        self._connection = {}
        self.result = self._compute_regex(frozenset(t for e in extracted for t in e.result))

    @lru_cache(maxsize=4096)
    def _compute_regex(self, tokens: Iterable[Tuple[str]]):

        tokenSet = set(tokens)

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

            lgroup = defaultdict(set)
            rgroup = defaultdict(set)
            lsegment = {}
            rsegment = {}
            for token in tokenSet:
                left = {token[i:]: token[:i] for i in range(1, len(token))}
                right = {v: k for k, v in left.items()}
                left[token] = right[token] = ()
                lsegment[token] = left
                rsegment[token] = right
                for k in left:
                    lgroup[k].add(token)
                for k in right:
                    rgroup[k].add(token)
            lgroup = {k: v for k, v in lgroup.items() if len(v) > 1}
            rgroup = {k: v for k, v in rgroup.items() if len(v) > 1}

            result = self._find_min_comb(tokenSet, lgroup, rgroup, lsegment, rsegment)

            result.sort()
            string = "|".join(result)

            if qmark or len(result) > 1:
                string = f"({string}){qmark}"

            return string

        elif len(tokenSet) > 1:
            char = sorted(i[0] for i in tokenSet)
            if ("-",) in tokenSet:
                char.append(char.pop(char.index("-")))
            if ("]",) in tokenSet:
                char.insert(0, char.pop(char.index("]")))
            return f'[{"".join(char)}]{qmark}'

        return f"{tokenSet.pop()[0]}{qmark}"

    def _find_min_comb(self, tokenSet: set, lgroup: dict, rgroup: dict, lsegment: dict, rsegment: dict):

        result = []

        if lgroup or rgroup:

            left = defaultdict(list)
            right = defaultdict(list)
            for group, segment, target in (lgroup, lsegment, left), (rgroup, rsegment, right):
                for k, v in group.items():
                    key = frozenset(segment[i][k] for i in v)
                    target[key].append(k)

            connection = self._connection
            connectionKeys = set()
            left = ((frozenset(j for i in v for j in lgroup[i]), (k, v)) for k, v in left.items())
            right = ((frozenset(j for i in v for j in rgroup[i]), (v, k)) for k, v in right.items())

            for i in (left, right):
                for key, val in i:
                    connectionKeys.add(key)
                    length = sum(len(j) for i in val for j in i)
                    try:
                        v = connection[key]
                    except KeyError:
                        connection[key] = [
                            val,  # Partition
                            sum(sum(len(i) for i in j) for j in key) + len(key) - 1,  # concatLength
                            length,  # Partition length
                            None,  # computed regex string
                            None,  # reduced length (concatLength - len(string))
                        ]
                    else:
                        if length < v[2]:
                            v[0] = val
                            v[2] = length
                            v[3] = v[4] = None

            compute_regex = self._compute_regex
            for group in self._group_keys(connectionKeys):

                mostReduced = -1
                optimal = None

                for key in group:
                    v = connection[key]
                    if v[1] <= mostReduced:
                        break

                    if v[3] is None:
                        left, right = (frozenset(i) for i in v[0])
                        v[3] = f"{compute_regex(left)}{compute_regex(right)}"
                        v[4] = v[1] - len(v[3])

                    if v[4] > mostReduced:
                        mostReduced = v[4]
                        optimal = key

                try:
                    result.append(connection[optimal][3])
                except KeyError:
                    continue

                subTokenSet = {j for i in group for j in i}
                tokenSet.difference_update(subTokenSet)
                subTokenSet.difference_update(optimal)
                if subTokenSet:
                    subLgroup = {k: i for k, v in lgroup.items() if len(i := v.intersection(subTokenSet)) > 1}
                    subRgroup = {k: i for k, v in rgroup.items() if len(i := v.intersection(subTokenSet)) > 1}
                    result.extend(self._find_min_comb(subTokenSet, subLgroup, subRgroup, lsegment, rsegment))

        if tokenSet:
            chars = frozenset(i for i in tokenSet if len(i) == 1)
            if chars:
                result.append(self._compute_regex(chars))
                tokenSet.difference_update(chars)
            result.extend(map("".join, tokenSet))

        return result

    def _group_keys(self, unVisited: set):
        """Group keys with common members together."""
        groups = []
        que = deque()

        while unVisited:
            currentVert = unVisited.pop()  # frozenset
            que.append(currentVert)
            currentGroup = [currentVert]
            groups.append(currentGroup)
            while que:
                currentVert = que.popleft()
                connected = tuple(i for i in unVisited if not currentVert.isdisjoint(i))
                unVisited.difference_update(connected)
                currentGroup.extend(connected)
                que.extend(connected)

        for group in groups:
            group.sort(key=lambda k: self._connection[k][1], reverse=True)
            yield group


def test_regex(regex: str, wordlist: list):
    extracted = Extractor(regex).get_text()
    assert sorted(wordlist) == sorted(extracted), "Extracted regex is different from original words."

    regex = re.compile(regex)
    for i in wordlist:
        if not regex.fullmatch(i):
            assert re.search(r"[*+{}]", i), "Regex matching test failed."