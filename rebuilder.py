#!/usr/bin/env python3

import argparse
import json
import os
import os.path as op
import pickle
import re
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import (ProcessPoolExecutor, ThreadPoolExecutor,
                                as_completed)
from itertools import chain, count, filterfalse, islice, repeat, tee
from operator import itemgetter
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, Union
from urllib.parse import urljoin, urlsplit

import requests
from lxml.etree import XPath
from lxml.html import HtmlElement
from lxml.html import fromstring as html_fromstring
from torrentool.api import Torrent

from regen import Regen

STDERR = sys.stderr
RE_G1 = r"[a-z]{3,10}"
RE_G2 = r"[0-9]{2,8}"
session = None


class LastPageReached(Exception):
    pass


class Scraper:

    ID_RE = rf"\s*({RE_G1})[_-]?({RE_G2})\s*"
    ID_FULLMATCH = True
    ex: ThreadPoolExecutor

    def get_id(self) -> Iterator[re.Match]:
        r = re.compile(self.ID_RE)
        r = r.fullmatch if self.ID_FULLMATCH else r.search
        return filter(None, map(r, map(str.lower, self._scrape_id())))

    def _scrape_id(self) -> Iterator[str]:
        raise NotImplementedError


class JavBusScraper(Scraper):

    STEP = 500
    XP = './/div[@id="waterfall"]//a[@class="movie-box"]//span/date[1]/text()'

    def __init__(self, ex):
        self.ex = ex
        self.xpath = xp_compile(self.XP)

    @staticmethod
    def get_tree(url: str, i: int):
        try:
            return get_tree(f"{url}{i}")
        except requests.HTTPError as e:
            if e.response.status_code == 404:
                raise LastPageReached(i)
            raise

    def _scrape(self,
                xpath: Union[XPath, str],
                domain: str,
                pages: Iterable[str],
                stop_null_page=False):

        print(f"Scanning {urlsplit(domain).netloc} ...", file=STDERR)
        if isinstance(xpath, str):
            xpath = xp_compile(xpath)

        for page in pages:
            lo = 1
            print(f"  {page}: ", end="", flush=True, file=STDERR)
            url = repeat(domain + page)
            try:
                while True:
                    hi = lo + self.STEP
                    print(f"{lo}..", end="", flush=True, file=STDERR)
                    trees = self.ex.map(self.get_tree, url, range(lo, hi))
                    yield from chain.from_iterable(map(xpath, trees))
                    lo = hi
            except LastPageReached as e:
                print(e, file=STDERR)
                if stop_null_page and e.args[0] == 1:
                    break

    def _scrape_id(self):
        return self._scrape(
            self.xpath,
            domain="https://www.javbus.com",
            pages=("/page/", "/uncensored/page/", "/genre/hd/",
                   "/uncensored/genre/hd/"),
        )

    def get_keyword(self):
        r = self._scrape(
            self.xpath,
            domain="https://www.javbus.org",
            pages=chain(("/page/",), (f"/studio/{i}/" for i in count(1))),
            stop_null_page=True,
        )
        f = re.compile(r"\s*([A-Za-z0-9]{3,})(?:\.\d\d){3}\s*").fullmatch
        return Counter(m[1].lower() for m in map(f, r) if m).items()


class JavDBScraper(JavBusScraper):

    STEP = 100
    XP = ('boolean(//nav[@class="pagination"]/ul[@class="pagination-list"]/li'
          '/a[contains(@class, "is-current") and number()=$page])')

    def get_tree(self, url: str, i: int):
        tree = get_tree(f"{url}?page={i}")
        if self.xpath(tree, page=i):
            return tree
        raise LastPageReached(i)

    def _scrape_id(self) -> Iterator[str]:
        return self._scrape(
            './/div[@id="videos"]//a[@class="box"]/div[@class="uid"]/text()',
            domain="https://javdb.com",
            pages=("/", "/uncensored"),
        )

    def get_keyword(self) -> Iterable[Tuple[str, int]]:
        r = self._scrape(
            '//div[@id="series"]//div[@class="box"]/a[@title and strong and span]',
            domain="https://javdb.com",
            pages=("/series/western",),
        )
        subspace = re.compile(r"\s+").sub
        retitle = re.compile(r"(?!\d+$)[a-z0-9]{3,}").fullmatch
        redigit = re.compile(r"\d+").search

        for a in r:
            title = subspace("", a.findtext("strong")).lower()
            if retitle(title):
                try:
                    yield title, int(redigit(a.findtext("span"))[0])
                except TypeError:
                    pass


class AVEScraper(Scraper):

    ID_RE = rf".+?:\s*({RE_G1})[_-]?({RE_G2})\s*"

    def __init__(self, ex) -> None:
        self.ex = ex

    def _scrape_id(self):

        url = "https://www.aventertainments.com/studiolists.aspx"
        print(f"Scanning {urlsplit(url).netloc} ...", file=STDERR)

        page_xp = xp_compile('string(//div[@class="pagination-rev"]'
                             '/ul/li[a/@title="Next"]'
                             '/preceding-sibling::li[1]/a/@href)')
        page_matcher = re.compile(r'(.*CountPage=)(\d+)(.*)', re.I).fullmatch
        id_xp = xp_compile(
            '//div[contains(@class, "single-slider-product--list")]'
            '/small/text()')
        submit = self.ex.submit

        tree = get_tree(url)
        url = tree.base_url
        url = frozenset(
            urljoin(url, u) for u in tree.xpath(
                '//div[contains(@class, "category-group-list")]'
                '/a/@href[contains(., "studio_products.aspx")]',
                smart_string=False))

        pool = []
        total = len(url)
        step = frozenset(range(1, total, (total // 10) or 1))
        print(f"  stage 1 ({total}): ", end="", file=STDERR)

        m = {"Rows": 3}
        url = as_completed(submit(get_tree, u, params=m) for u in url)
        for i, tree in enumerate(url, 1):
            if i in step:
                print(f"{i}..", end="", flush=True, file=STDERR)
            tree = tree.result()
            m = page_matcher(page_xp(tree))
            if m:
                url = f"{urljoin(tree.base_url, m[1])}{{}}{m[3]}".format
                m = range(2, int(m[2]) + 1)
                pool.extend(submit(get_tree, url(i)) for i in m)
            yield from id_xp(tree)
        print(total, file=STDERR)

        total = len(pool)
        step = frozenset(range(1, total, (total // 10) or 1))
        print(f"  stage 2 ({total}): ", end="", file=STDERR)

        pool = as_completed(pool)
        for i, tree in enumerate(pool, 1):
            if i in step:
                print(f"{i}..", end="", flush=True, file=STDERR)
            yield from id_xp(tree.result())
        print(total, file=STDERR)


class DMMScraper(Scraper):

    ID_RE = rf"/cid=(?:[a-z]+_)?\d*({RE_G1})({RE_G2})[a-z]?/"
    ID_FULLMATCH = False

    def __init__(self, ex) -> None:
        self.ex = ex

    def _scrape_id(self):

        url = (
            "https://www.dmm.co.jp/digital/videoa/-/list/=/sort=release_date/view=text/",
            "https://www.dmm.co.jp/digital/videoc/-/list/=/sort=release_date/view=text/",
        )
        print(f"Scanning {urlsplit(url[0]).netloc} ...", file=STDERR)

        submit = self.ex.submit
        xpath = xp_compile('.//div[@class="d-area"]//div[@class="d-item"]'
                           '//tr/td[1]/p[@class="ttl"]/a/@href')

        pool = [submit(get_tree, u) for u in url]
        for ft in as_completed(pool):
            tree = ft.result()
            i = tree.xpath(
                'string(.//div[@class="list-capt"]//li[@class="terminal"]'
                '/a/@href[contains(., "/page=")])')
            i = re.fullmatch(r"(.*/page=)(\d+)(/.*)", urljoin(tree.base_url, i))
            url = f"{i[1]}{{}}{i[3]}".format
            i = range(2, int(i[2]) + 1)
            pool.extend(submit(get_tree, url(j)) for j in i)

        total = len(pool)
        step = frozenset(range(1, total, (total // 10) or 1))
        print(f"  total ({total}): ", end="", flush=True, file=STDERR)

        pool = as_completed(pool)
        for i, ft in enumerate(pool, 1):
            if i in step:
                print(f"{i}..", end="", flush=True, file=STDERR)
            yield from xpath(ft.result())
        print(total, file=STDERR)


class MGSJsonLoader(Scraper):

    ID_RE = rf"\d*({RE_G1})[_-]?({RE_G2})"

    def __init__(self, path) -> None:
        self.path = path

    def _scrape_id(self) -> Iterable[str]:
        path = self.path
        if not path:
            return ()
        print(f"Loading {op.basename(path)} ...", file=STDERR)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


class MTeamCollector:

    DOMAIN = "https://pt.m-team.cc"

    def __init__(self, *, username: str, password: str, page_max: int,
                 cache_dir: str, av_page: str, non_av_page: str) -> None:

        self._account = {"username": username, "password": password}
        self._page_max = page_max
        self._cache_dirs = (
            op.join(cache_dir, "non_av"),
            op.join(cache_dir, "av"),
        )
        self._urls = (
            urljoin(self.DOMAIN, non_av_page),
            urljoin(self.DOMAIN, av_page),
        )
        for cache_dir in self._cache_dirs:
            os.makedirs(cache_dir, exist_ok=True)

    def from_cache(self, is_av: bool) -> Iterator[str]:
        matcher = re.compile(r"[0-9]+\.txt").fullmatch
        with os.scandir(self._cache_dirs[is_av]) as it:
            for entry in it:
                if matcher(entry.name):
                    yield entry.path

    def from_web(self, is_av: bool) -> Iterator[str]:

        url = self._urls[is_av]
        cache_dir = self._cache_dirs[is_av]
        pool = {}
        idx = 0
        total = self._page_max
        join = op.join
        exists = op.exists
        step = frozenset(range(1, total, (total // 10) or 1))
        matcher = re.compile(r"\bid=([0-9]+)").search
        xpath = xp_compile(
            '//form[@id="form_torrent"]//table[@class="torrentname"]'
            '/descendant::a[contains(@href, "download.php?")][1]/@href')

        print(f"Scanning mteam ({total} pages)...",
              end="",
              flush=True,
              file=STDERR)
        self._login()

        with ThreadPoolExecutor() as ex:

            for ft in as_completed(
                    ex.submit(get_tree, url, params={"page": i})
                    for i in range(1, total + 1)):

                idx += 1
                if idx in step:
                    print(f"{idx}..", end="", flush=True, file=STDERR)

                for url in xpath(ft.result()):
                    try:
                        path = join(cache_dir, matcher(url)[1] + ".txt")
                    except TypeError:
                        continue
                    if exists(path):
                        yield path
                    else:
                        url = urljoin(self.DOMAIN, url)
                        pool[ex.submit(get_response, url)] = url, path
            print(total, file=STDERR)

            idx = 0
            total = len(pool)
            fmt = f"[{{:{len(str(total))}d}}/{total}] {{}}".format

            for ft in as_completed(pool):
                url, path = pool[ft]
                idx += 1
                print(fmt(idx, url), file=STDERR)
                try:
                    self._parse_torrent(ft.result().content, path)
                except requests.RequestException as e:
                    print(e, file=STDERR)
                except Exception as e:
                    print(getattr(e, "stderr", "").strip() or e, file=STDERR)
                    try:
                        os.unlink(path)
                    except OSError:
                        pass
                else:
                    yield path

    def _login(self):
        try:
            r = session.head(self._urls[0], allow_redirects=True)
            r.raise_for_status()
            if "/login.php" in r.url:
                print("login...", end="", flush=True)
                session.post(
                    url=self.DOMAIN + "/takelogin.php",
                    data=self._account,
                    headers={"referer": self.DOMAIN + "/login.php"},
                )
                r = session.head(self._urls[0], allow_redirects=True)
                r.raise_for_status()
                if "/login.php" in r.url:
                    raise requests.RequestException("invalid credentials")
            self._login = self._skip
        except requests.RequestException as e:
            sys.exit(f"login failed: {e}")

    def _parse_torrent(self, content: bytes, path: str):
        """Parse a torrent, write file list to `path`."""
        try:
            files = Torrent.from_string(content).files
            with open(path, "w", encoding="utf-8") as f:
                f.writelines(i[0].lower() + "\n" for i in files)
        except OSError:
            raise
        except Exception as e:
            self._parse_torrent2(e, content, path)

    def _parse_torrent2(self, e: Exception, content: bytes, path: str):

        torrent_file = path + ".torrent"
        try:
            with open(torrent_file, "wb") as f:
                f.write(content)
            files = subprocess.run(
                ("transmission-show", torrent_file),
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        except FileNotFoundError:
            print(
                "Error: transmission-show not found. It is recommended "
                "to install transmission-show to handle more torrents.\n"
                "In Ubuntu, try: 'sudo apt install transmission-cli'",
                file=STDERR)
            self._parse_torrent2 = self._skip
            raise e
        finally:
            try:
                os.unlink(torrent_file)
            except OSError:
                pass

        spliter = re.compile(r"^\s+(.+) \([^)]+\)$", flags=re.M)
        files = spliter.findall(files, files.index("\n\nFILES\n\n"))
        if not files:
            raise ValueError("torrent file seems empty")
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(i.lower() + "\n" for i in files)

    @staticmethod
    def _skip(e=None, *args):
        if e is not None:
            raise e


class Builder:

    def __init__(self, *, regex_file: str, keyword_max: int, prefix_max: int,
                 mgs_json: str, **kwargs) -> None:

        self._regex_file = regex_file
        self._keyword_max = keyword_max
        self._prefix_max = prefix_max
        self._mgs_json = mgs_json
        self._datafile = "data.json"

    def from_cache(self) -> Optional[str]:
        try:
            with open(self._datafile, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError) as e:
            sys.exit(e)
        return self._build(data)

    def from_web(self) -> Optional[str]:

        with ThreadPoolExecutor() as ex:
            scrapers = (
                JavBusScraper(ex),
                JavDBScraper(ex),
                AVEScraper(ex),
                DMMScraper(ex),
                MGSJsonLoader(self._mgs_json),
            )
            data = {
                "prefix": self._scrape_prefix(scrapers),
                "keyword": self._scrape_keyword(scrapers, ex),
            }
        with open(self._datafile, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        return self._build(data)

    def _scrape_prefix(self, scrapers: Tuple[Scraper]) -> Dict[str, int]:
        d = defaultdict(set)
        for s in scrapers:
            for m in s.get_id():
                d[m[1]].add(int(m[2]))
        return self._sort_scrape(zip(d, map(len, d.values())))

    def _scrape_keyword(self, scrapers, ex) -> Dict[str, int]:
        d = {}
        setdefault = d.setdefault
        freq = ex.submit(get_freqwords)
        for s in scrapers:
            if not hasattr(s, "get_keyword"):
                continue
            for k, v in s.get_keyword():
                if setdefault(k, v) < v:
                    d[k] = v
        for k in freq.result().intersection(d):
            del d[k]
        return self._sort_scrape(d.items())

    @staticmethod
    def _sort_scrape(d: Iterable[Tuple[str, int]]):
        d = sorted(d)
        d.sort(key=itemgetter(1), reverse=True)
        return dict(d)

    def _build(self, data: Dict[str, dict]):

        keyword = self._build_regex(
            name="keyword",
            data=data,
            omitOuterParen=True,
        )
        prefix = self._build_regex(
            name="prefix",
            data=data,
            omitOuterParen=False,
            filterlist=(keyword,),
        )

        print("-" * 50, file=STDERR)
        if not (keyword and prefix):
            print("Generating regex failed.", file=STDERR)
            return

        regex = rf"(^|[^a-z0-9])({keyword}|[0-9]{{,3}}{prefix}[_-]?{RE_G2})([^a-z0-9]|$)"
        self._update_file(self._regex_file, regex)

        print(f"Result: {len(regex)} chars", file=STDERR)
        print(regex)
        return regex

    def _build_regex(self,
                     name: str,
                     data: Dict[str, dict],
                     omitOuterParen: bool,
                     filterlist: Tuple[str] = None) -> str:

        print(f" {name.upper()} ".center(50, "-"), file=STDERR)
        data = data[name]
        total = sum(data.values())
        print(f"Entry: {total}, {name}: {len(data)}", file=STDERR)

        words = sorted(data, key=data.get, reverse=True)
        lo = getattr(self, f"_{name}_max")
        hi = len(words)
        if 0 < lo < hi:
            x = data[words[lo]]
            while lo < hi:
                mid = (lo + hi) // 2
                if x > data[words[mid]]:
                    hi = mid
                else:
                    lo = mid + 1
            words = words[:lo]
        if not words:
            return

        print("Cut: {}, frequency: {}, coverage: {:.1%}".format(
            len(words), data[words[-1]],
            sum(map(data.get, words)) / total),
              file=STDERR)

        whitelist = self._update_file(name + "_whitelist.txt")
        blacklist = self._update_file(name + "_blacklist.txt")
        regex = chain(whitelist, blacklist, filterlist or ())
        regex = re.compile("|".join(regex)).fullmatch
        words[:] = filterfalse(regex, words)
        words.extend(whitelist)
        words.sort()
        print(f"Final: {len(words)}", file=STDERR)

        regen = Regen(words)
        regex = regen.to_regex(omitOuterParen=omitOuterParen)
        concat = "|".join(words)
        if not omitOuterParen and len(words) > 1:
            concat = f"({concat})"

        length = len(regex)
        diff = length - len(concat)
        if diff > 0:
            print(
                f"Computed regex is {diff} characters "
                "longer than concatenation, use the latter.",
                file=STDERR)
            regex = concat
        else:
            regen.raise_for_verify()
            print(f"Regex length: {length} ({diff})", file=STDERR)

        return regex

    def _update_file(self, file: str, content: str = None):

        new = () if content is None else [content]
        try:
            with open(file, "r+", encoding="utf-8") as f:
                old = f.read().splitlines()
                if not new:
                    new = self._sort_custom(old)
                if old != new:
                    f.seek(0)
                    f.writelines(i + "\n" for i in new)
                    f.truncate()
                    print(f"Update: {file}", file=STDERR)
        except FileNotFoundError:
            os.makedirs(op.dirname(file), exist_ok=True)
            with open(file, mode="w", encoding="utf-8") as f:
                f.writelines(i + "\n" for i in new)
            print(f"Create: {file}", file=STDERR)
        return new

    @staticmethod
    def _sort_custom(a: List[str]):
        return sorted(set(map(str.lower, filter(None, map(str.strip, a)))))


class Analyzer:

    def __init__(self, *, regex_file: str, mteam: dict, **kwargs) -> None:

        try:
            with open(regex_file, "r", encoding="utf-8") as f:
                regex = f.readline().strip()
                if f.read():
                    raise ValueError("regex file should contain only one line")
            if "(?" in regex:
                raise ValueError("regex should have no special groups")
            i = regex.index("(", 1)
            regex = "{}({}".format(regex[:i].replace("(", "(?:"),
                                   regex[i + 1:].replace("(", "(?:"))
            self.mt = re.compile(regex, flags=re.M).search
        except (OSError, ValueError) as e:
            sys.exit(e)

        self.ext = re.compile(
            r"\.(?:m(?:p4|[24kop]v|2?ts|4p|p2|pe?g|xf)|wmv|"
            r"avi|iso|3gp|asf|bdmv|flv|rm|rmvb|ts|vob|webm)$",
            flags=re.M).search
        self.regex_file = regex_file
        self._mteam = MTeamCollector(**mteam)

    def analyze_av(self, local: bool = False):

        print("Matching test begins with av torrents...", file=STDERR)

        report_file = op.abspath("av_report.txt")
        raw_file = op.abspath("mismatch_raw.txt")
        total = count = 0
        strings = defaultdict(set)
        prefix_count = Counter()
        word_count = Counter()
        tmp = set()
        prefix_finder = re.compile(
            r"(?:^|/)([0-9]{,3}([a-z]{2,10})"
            r"(?:-[0-9]{2,8}|[0-9]{2,8}[hm]hb[0-9]{,2}))\b.*$",
            flags=re.M).findall
        word_finder = re.compile(r"(?:\b|_)([a-z]{3,})(?:\b|_)", re.M).findall

        paths = self._mteam.from_cache if local else self._mteam.from_web
        paths = paths(is_av=True)

        with ProcessPoolExecutor() as ex, \
             open(raw_file, "w", encoding="utf-8") as f:

            freqwords = ex.submit(get_freqwords)
            for content in ex.map(self._match_av, paths, chunksize=100):
                total += 1
                if content:
                    count += 1
                    f.write(content)
                    f.write("---\n")
                    for string, prefix in prefix_finder(content):
                        strings[prefix].add(string)
                        tmp.add(prefix)
                    prefix_count.update(tmp)
                    tmp.clear()

                    tmp.update(word_finder(content))
                    word_count.update(tmp)
                    tmp.clear()
            freqwords = freqwords.result()

        prefix_count = [
            (i, k, strings[k]) for k, i in prefix_count.items() if i >= 5
        ]
        word_count = [(i, k)
                      for k, i in word_count.items()
                      if i >= 5 and k not in freqwords]

        f = lambda t: (-t[0], t[1])
        prefix_count.sort(key=f)
        word_count.sort(key=f)
        report = self._format_report(total, total - count,
                                     "Potential ID Prefixes", prefix_count)

        with open(report_file, "w", encoding="utf-8") as f:
            for line in report:
                print(line, end="")
                f.write(line)

            f.write("\n\nPotential Keywords:\n"
                    f'{"torrent":>7}  word\n'
                    f'{"-" * 80}\n')
            f.writelines(f"{i:7d}  {j}\n" for i, j in word_count)

        print(f"Result saved to: {report_file}", file=STDERR)

    def analyze_nonav(self, local: bool = False):

        print("Matching test begins with non-av torrents...", file=STDERR)

        report_file = op.abspath("nonav_report.txt")
        total = count = 0
        searcher = re.compile(r"[a-z]+").search

        strings = defaultdict(set)
        word_count = Counter()
        tmp = set()
        paths = self._mteam.from_cache if local else self._mteam.from_web
        paths = paths(is_av=False)

        with ProcessPoolExecutor() as ex:
            for video in ex.map(self._match_nonav, paths, chunksize=100):
                total += 1
                if video:
                    count += 1
                    for string in video:
                        try:
                            word = searcher(string)[0]
                        except TypeError:
                            word = string
                        strings[word].add(string)
                        tmp.add(word)
                    word_count.update(tmp)
                    tmp.clear()

        word_count = [(i, k, strings[k]) for k, i in word_count.items()]
        word_count.sort(key=lambda t: (-t[0], t[1]))
        report = self._format_report(total, count, "Matched Strings",
                                     word_count)

        with open(report_file, "w", encoding="utf-8") as f:
            for line in report:
                print(line, end="")
                f.write(line)
        print(f"Result saved to: {report_file}", file=STDERR)

    def _match_av(self, path: str) -> Optional[str]:
        with open(path, "r", encoding="utf-8") as f:
            a, b = tee(filter(self.ext, f))
            if not any(map(self.mt, a)):
                return "".join(b)

    def _match_nonav(self, path: str) -> Tuple[str]:
        with open(path, "r", encoding="utf-8") as f:
            return tuple(m[1] for m in map(self.mt, filter(self.ext, f)) if m)

    def _format_report(self, total, count, title, result):
        f = self._slice_on_len
        yield (
            f"Regex file: {self.regex_file}\n"
            f"Total: {total}, Matched: {count}, Percentage: {count / total:.2%}\n\n"
            f"{title}:\n"
            f'{"torrent":>7}  {"word":15} strings\n{"-" * 80}\n')
        for i, k, s in result:
            yield f'{i:7d}  {k:15} {", ".join(f(s))}\n'

    @staticmethod
    def _slice_on_len(a: Iterable[str], n: int = 80):
        i = 0
        for x in a:
            i += len(x) + 2
            if i >= n:
                yield "..."
                break
            yield x


def init_session(path: str):

    from urllib3 import Retry

    global session
    session = requests.Session()
    session.headers.update({
        "User-Agent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                      'AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/88.0.4324.104 Safari/537.36'
    })
    adapter = requests.adapters.HTTPAdapter(
        max_retries=Retry(total=7,
                          status_forcelist=frozenset((500, 502, 503, 504)),
                          backoff_factor=0.3))
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    try:
        with open(path, "rb") as f:
            session.cookies = pickle.load(f)
    except FileNotFoundError:
        create_cookie = requests.cookies.create_cookie
        session.cookies.set_cookie(
            create_cookie(
                domain="www.javbus.com",
                name="existmag",
                value="all",
            ))
        session.cookies.set_cookie(
            create_cookie(
                domain="dmm.co.jp",
                name="age_check_done",
                value="1",
            ))


def get_response(url: str, **kwargs) -> requests.Response:
    response = session.get(url, timeout=(9.1, 60), **kwargs)
    response.raise_for_status()
    return response


def get_tree(url: str, **kwargs) -> HtmlElement:
    response = session.get(url, timeout=(9.1, 60), **kwargs)
    response.raise_for_status()
    return html_fromstring(response.content, base_url=response.url)


def get_freqwords(lo=3, k: int = 3000):
    """Get wordlist of the top `k` English words longer than `lo` letters."""
    u = "https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english-usa.txt"
    m = re.finditer(rf"^\s*([A-Za-z]{{{lo},}})\s*$", get_response(u).text, re.M)
    return frozenset(map(str.lower, map(itemgetter(1), islice(m, k))))


def dump_cookies(path: str):
    with open(path, "wb") as f:
        pickle.dump(session.cookies, f)


def xp_compile(path: str):
    return XPath(path, regexp=False, smart_strings=False)


def parse_config(configfile: str) -> dict:
    try:
        with open(configfile, "r", encoding="utf-8") as f:
            config = json.load(f)
        a = op.normpath
        b = op.expanduser
        config["regex_file"] = a(b(config["regex_file"]))
        config["mgs_json"] = a(b(config["mgs_json"]))
        config["mteam"]["cache_dir"] = a(b(config["mteam"]["cache_dir"]))
    except FileNotFoundError:
        pass
    except Exception as e:
        sys.exit(f"Error in config file: {e}")
    else:
        return config

    default = {
        "regex_file": "regex.txt",
        "keyword_max": 150,
        "prefix_max": 3000,
        "mgs_json": "",
        "mteam": {
            "username": "",
            "password": "",
            "page_max": 500,
            "cache_dir": "cache",
            "av_page": "/adult.php?cat410=1&cat429=1&cat424=1&cat430=1&cat426=1&cat437=1&cat431=1&cat432=1",
            "non_av_page": "/torrents.php",
        },
    } # yapf: disable

    with open(configfile, "w", encoding="utf-8") as f:
        json.dump(default, f, indent=4)
    sys.exit(f"Please edit {configfile} before running me again.")


def parse_arguments():

    parser = argparse.ArgumentParser(
        description="The Ultimate Regex Builder, by David Pi.")

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-b",
        "--build",
        dest="mode",
        action="store_const",
        const="build",
        help="generate regex and save to file (default)",
    )
    group.add_argument(
        "-t",
        "--test-av",
        dest="mode",
        action="store_const",
        const="test_av",
        help="test regex with av torrents",
    )
    group.add_argument(
        "-m",
        "--test-nonav",
        dest="mode",
        action="store_const",
        const="test_nonav",
        help="test regex with non-av torrents",
    )
    group.set_defaults(mode="build")
    parser.add_argument(
        "-l",
        "--local",
        dest="local",
        action="store_true",
        help="use cached data instead of web scraping (default %(default)s)",
    )

    group = parser.add_argument_group(title="configfile override")
    group.add_argument(
        "--file",
        dest="file",
        action="store",
        help="the target file, override 'regex_file'",
    )
    group.add_argument(
        "--kmax",
        dest="keyword_max",
        action="store",
        type=int,
        help="maximum keywords, override 'keyword_max' (0 for unlimited)",
    )
    group.add_argument(
        "--pmax",
        dest="prefix_max",
        action="store",
        type=int,
        help="maximum prefixes, override 'prefix_max' (0 for unlimited)",
    )
    group.add_argument(
        "--mtmax",
        dest="mteam_max",
        action="store",
        type=int,
        help="maximum mteam pages to scan, override 'mteam.page_max'",
    )
    return parser.parse_args()


def main():

    args = parse_arguments()

    config = op.join(op.dirname(__file__), "builder")
    try:
        os.chdir(config)
    except FileNotFoundError:
        os.mkdir(config)
        os.chdir(config)
    config = parse_config("config.json")
    init_session("cookies")

    if args.file:
        config["regex_file"] = args.file

    if args.mode == "build":

        if args.keyword_max is not None:
            config["keyword_max"] = args.keyword_max
        if args.prefix_max is not None:
            config["prefix_max"] = args.prefix_max

        builder = Builder(**config)
        if args.local:
            builder.from_cache()
        else:
            builder.from_web()

    else:

        if args.mteam_max is not None:
            config["mteam"]["page_max"] = args.mteam_max

        analyzer = Analyzer(**config)
        if args.mode == "test_av":
            analyzer.analyze_av(args.local)
        else:
            analyzer.analyze_nonav(args.local)

    dump_cookies("cookies")


if __name__ == "__main__":
    main()
