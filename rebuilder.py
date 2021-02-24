#!/usr/bin/env python3

import argparse
import json
import os
import os.path as op
import pickle
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import (ProcessPoolExecutor, ThreadPoolExecutor,
                                as_completed)
from itertools import chain, count, filterfalse, islice, repeat, tee
from operator import itemgetter
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, Union
from urllib.parse import urljoin

import requests
from lxml.etree import XPath
from lxml.html import HtmlElement
from lxml.html import fromstring as html_fromstring

from regen import Regen

try:
    from bencoder import bdecode
except ImportError:
    bdecode = None

STDERR = sys.stderr
JAV_RE = r"([a-z]{3,10})[_-]?([0-9]{2,8})[abcrz]?"
session = None


class LastPageReached(Exception):
    pass


class Scraper:
    """Base class for all scrapers."""

    __slots__ = "ex"
    ID_RE = JAV_RE
    DATA_RE: str

    def __init__(self, ex: ThreadPoolExecutor) -> None:
        self.ex = ex

    def get_id(self) -> Iterator[re.Match]:

        print(f"Scanning {self.scraper_name} for product ids...", file=STDERR)

        try:
            data = map(re.compile(self.DATA_RE).search, self._scrape_id())
            data = sorted(frozenset(map(itemgetter(1), filter(None, data))))
            if not data:
                raise ValueError("empty result")
        except Exception as e:
            try:
                with open(self.jsonfile, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (OSError, ValueError):
                raise e
            else:
                print(f"Scrapper error, use cache: {e}", file=STDERR)
        else:
            with open(self.jsonfile, "w", encoding="utf-8") as f:
                json.dump(data, f, separators=(",", ":"))

        print(f"  Entries: {len(data)}", file=STDERR)
        f = re.compile(self.ID_RE).fullmatch
        return filter(None, map(f, map(str.lower, data)))

    def _scrape_id(self) -> Iterator[str]:
        raise NotImplementedError

    @property
    def scraper_name(self):
        return self.__class__.__name__.rpartition("Scraper")[0]

    @property
    def jsonfile(self):
        return op.join("data", self.scraper_name.lower() + ".json")


class JavBusScraper(Scraper):

    __slots__ = "xpath"
    DATA_RE = r"^\s*([A-Za-z0-9_-]+)\s*$"
    STEP = 500
    XP = './/div[@id="waterfall"]//a[@class="movie-box"]//span/date[1]/text()'

    def __init__(self, ex: ThreadPoolExecutor) -> None:
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
                stop_null_page: bool = False):

        if isinstance(xpath, str):
            xpath = xp_compile(xpath)
        write = STDERR.write

        for page in pages:
            i = 1
            url = repeat(domain + page)
            try:
                while True:
                    for t in self.ex.map(
                            self.get_tree,
                            url,
                            range(i, i + self.STEP),
                    ):
                        write(f'  {page}: [{i}] 8={"=" * (i // 50)}Э\r')
                        i += 1
                        yield from xpath(t)
            except LastPageReached:
                if i > 1:
                    write("\n")
                elif stop_null_page:
                    break

    def _scrape_id(self):
        return self._scrape(
            self.xpath,
            domain="https://www.javbus.com",
            pages=("/page/", "/uncensored/page/", "/genre/hd/",
                   "/uncensored/genre/hd/"),
        )

    def get_keyword(self):

        print(f"Scanning {self.scraper_name} for keywords...", file=STDERR)
        r = self._scrape(
            self.xpath,
            domain="https://www.javbus.org",
            pages=chain(("/page/",), (f"/studio/{i}/" for i in count(1))),
            stop_null_page=True,
        )
        f = re.compile(r"\s*([A-Za-z0-9]{3,})(?:\.\d\d){3}\s*").fullmatch
        return Counter(m[1].lower() for m in map(f, r) if m).items()


class JavDBScraper(JavBusScraper):

    __slots__ = ()
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

        print(f"Scanning {self.scraper_name} for keywords...", file=STDERR)
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

    __slots__ = ()
    DATA_RE = r"^.+?:\s*([A-Za-z0-9_-]+)\s*$"

    def _scrape_id(self):

        url = "https://www.aventertainments.com/studiolists.aspx"
        tree = get_tree(url)
        url = tree.base_url
        url = frozenset(
            urljoin(url, u) for u in tree.xpath(
                '//div[contains(@class, "category-group-list")]'
                '/a/@href[contains(., "studio_products.aspx")]',
                smart_string=False))

        pool = []
        total = len(url)
        submit = self.ex.submit
        m = {"Rows": 3}
        url = as_completed(submit(get_tree, u, params=m) for u in url)
        re_page = re.compile(r"(.*CountPage=)(\d+)(.*)", re.I).fullmatch
        xp_page = xp_compile(
            'string(//div[@class="pagination-rev"]/ul/li[a/@title="Next"]'
            '/preceding-sibling::li[1]/a/@href)')
        xpath = xp_compile(
            '//div[contains(@class, "single-slider-product--list")]'
            '/small/text()')

        for tree in progress(url, total, prefix="Step 1"):
            tree = tree.result()
            m = re_page(xp_page(tree))
            if m:
                url = f"{urljoin(tree.base_url, m[1])}{{}}{m[3]}".format
                m = range(2, int(m[2]) + 1)
                pool.extend(submit(get_tree, url(i)) for i in m)
            yield from xpath(tree)

        total = len(pool)
        pool = as_completed(pool)
        for tree in progress(pool, total, prefix="Step 2"):
            yield from xpath(tree.result())


class DMMScraper(Scraper):

    __slots__ = ()
    DATA_RE = r"/cid=([A-Za-z0-9_-]+)/"
    ID_RE = rf"(?:[a-z]+_)?\d*{JAV_RE}"

    def _scrape_id(self):

        url = (
            "https://www.dmm.co.jp/digital/videoa/-/list/=/sort=release_date/view=text/",
            "https://www.dmm.co.jp/digital/videoc/-/list/=/sort=release_date/view=text/",
        )
        submit = self.ex.submit
        pool = {submit(get_tree, u) for u in url}
        for ft in as_completed(pool):
            tree = ft.result()
            total = tree.xpath(
                'string(.//div[@class="list-capt"]//li[@class="terminal"]'
                '/a/@href[contains(., "/page=")])')
            total = re.fullmatch(r"(.*/page=)(\d+)(/.*)",
                                 urljoin(tree.base_url, total))
            url = f"{total[1]}{{}}{total[3]}".format
            total = range(2, int(total[2]) + 1)
            pool.update(submit(get_tree, url(i)) for i in total)

        total = len(pool)
        pool = as_completed(pool)
        xpath = xp_compile('.//div[@class="d-area"]//div[@class="d-item"]'
                           '//tr/td[1]/p[@class="ttl"]/a/@href')
        for ft in progress(pool, total):
            yield from xpath(ft.result())


class MGSScraper(Scraper):

    __slots__ = ()
    DATA_RE = r"product_detail/([A-Za-z0-9_-]+)/?$"
    ID_RE = rf"\d*{JAV_RE}"

    def _scrape_id(self):

        url = "https://www.mgstage.com/ppv/makers.php?id=osusume"
        submit = self.ex.submit
        tree = get_tree(url)
        url = tree.base_url
        results = tree.xpath('//div[@id="maker_list"]/dl[@class="navi"]'
                             '/dd/a/@href[contains(., "makers.php")]')
        results = {urljoin(url, u) for u in results}
        results.discard(url)

        total = len(results) + 1
        results = chain(self.ex.map(get_tree, results), (tree,))
        pool = {}
        xpath = xp_compile(
            '//div[@id="maker_list"]/div[@class="maker_list_box"]'
            '/dl/dt/a[2]/@href[contains(., "search.php")]')
        for tree in progress(results, total, prefix="Step 1"):
            url = tree.base_url
            for m in xpath(tree):
                m = urljoin(url, m)
                if m not in pool:
                    pool[m] = submit(get_tree, m)

        total = len(pool)
        results = as_completed(pool.values())
        pool = []
        re_page = re.compile(r"(.*page=)(\d+)(.*)").fullmatch
        xp_page = xp_compile('string(//div[@class="pager_search_bottom"]'
                             '//a[contains(., "最後")]/@href)')
        xpath = xp_compile('//article[@id="center_column"]'
                           '//div[@class="rank_list"]//li/h5/a/@href')
        for tree in progress(results, total, prefix="Step 2"):
            tree = tree.result()
            m = re_page(xp_page(tree))
            if m:
                url = f"{urljoin(tree.base_url, m[1])}{{}}{m[3]}".format
                m = int(m[2]) + 1
                pool.extend(submit(get_tree, url(i)) for i in range(2, m))
            yield from xpath(tree)

        total = len(pool)
        results = as_completed(pool)
        del pool
        for tree in progress(results, total, prefix="Step 3"):
            yield from xpath(tree.result())


class Builder:

    def __init__(self, *, regex_file: str, keyword_max: int, prefix_max: int,
                 **kwargs) -> None:

        self._regex_file = regex_file
        self._keyword_max = keyword_max
        self._prefix_max = prefix_max
        self._datafile = op.join("data", "frequency.json")

    def from_cache(self) -> Optional[str]:
        try:
            with open(self._datafile, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError) as e:
            sys.exit("Loading local cache failed. A full "
                     f"web scrape may fix the problem.\n{e}")
        return self._build(data)

    def from_web(self) -> Optional[str]:

        with ThreadPoolExecutor() as ex:
            scrapers = (
                JavBusScraper(ex),
                JavDBScraper(ex),
                AVEScraper(ex),
                DMMScraper(ex),
                MGSScraper(ex),
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
            if hasattr(s, "get_keyword"):
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

        regex = (
            r"(^|[^a-z0-9])"
            rf"({keyword}|[0-9]{{,5}}{prefix}[_-]?[0-9]{{2,8}}([abcrz]|f?hd)?)"
            r"([^a-z0-9]|$)")
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
            with open(file, "r+", encoding="utf-8", newline="\n") as f:
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
            with open(file, mode="w", encoding="utf-8", newline="\n") as f:
                f.writelines(i + "\n" for i in new)
            print(f"Create: {file}", file=STDERR)
        return new

    @staticmethod
    def _sort_custom(a: List[str]) -> List[str]:
        return sorted(set(map(str.lower, filter(None, map(str.strip, a)))))


class MTeamCollector:

    DOMAIN = "https://pt.m-team.cc"

    def __init__(self, *, username: str, password: str, page_max: int,
                 cache_dir: str, av_page: str, non_av_page: str) -> None:

        if bdecode is None:
            sys.exit("Error: require module 'bencoder.pyx'")

        self._account = {"username": username, "password": password}
        self._page_max = page_max
        self._cachedirs = (
            op.join(cache_dir, "non_av"),
            op.join(cache_dir, "av"),
        )
        self._urls = (
            urljoin(self.DOMAIN, non_av_page),
            urljoin(self.DOMAIN, av_page),
        )
        for cache_dir in self._cachedirs:
            os.makedirs(cache_dir, exist_ok=True)

    def from_cache(self, is_av: bool) -> Iterator[str]:
        matcher = re.compile(r"[0-9]+\.txt").fullmatch
        with os.scandir(self._cachedirs[is_av]) as it:
            for entry in it:
                if matcher(entry.name):
                    yield entry.path

    def from_web(self, is_av: bool) -> Iterator[str]:

        print(f"Scanning mteam...", file=STDERR)
        cachedir = self._cachedirs[is_av]
        pool = {}
        join = op.join
        exists = op.exists
        matcher = re.compile(r"\bid=([0-9]+)").search

        with ThreadPoolExecutor() as ex:

            for url in self._get_links(self._urls[is_av], ex):
                try:
                    path = join(cachedir, matcher(url)[1] + ".txt")
                except TypeError:
                    continue
                if exists(path):
                    yield path
                else:
                    url = urljoin(self.DOMAIN, url)
                    pool[ex.submit(get_response, url)] = url, path

            i = str(len(pool))
            fmt = f"  [{{:{len(i)}d}}/{i}] {{}}".format
            for i, ft in enumerate(as_completed(pool), 1):
                url, path = pool.pop(ft)
                print(fmt(i, url), file=STDERR)
                try:
                    self._parse_torrent(ft.result().content, path)
                except requests.RequestException as e:
                    print(e, file=STDERR)
                except Exception as e:
                    print(f"Error parsing torrent: {e}", file=STDERR)
                    try:
                        os.unlink(path)
                    except OSError:
                        pass
                else:
                    yield path

    def _get_links(self, url: str, ex: ThreadPoolExecutor) -> Iterator[str]:
        """Scan pages up to _page_max, yields download links."""

        # mteam page indexes start at 0, display numbers should start at 1
        tree = self._login(url, params={"page": 0})
        total = tree.xpath(
            'string(//td[@id="outer"]/table//td/p[@align="center"]'
            '/a[contains(@href, "page=")][last()]/@href)')
        total = int(re.search(r"\bpage=(\d+)", total)[1]) + 1
        if 0 < self._page_max < total:
            total = self._page_max

        pool = as_completed({
            ex.submit(get_tree, url, params={"page": i})
            for i in range(1, total)
        })
        xpath = xp_compile(
            '//form[@id="form_torrent"]//table[@class="torrentname"]'
            '/descendant::a[contains(@href, "download.php?")][1]/@href')
        yield from xpath(tree)

        for tree in progress(pool, total, 2):
            yield from xpath(tree.result())

    def _login(self, url, **kwargs):
        tree = get_tree(url, **kwargs)
        if "/login.php" in tree.base_url:
            print("Login...", end="", flush=True, file=STDERR)
            session.post(
                url=self.DOMAIN + "/takelogin.php",
                data=self._account,
                headers={"referer": self.DOMAIN + "/login.php"},
            )
            tree = get_tree(url, **kwargs)
            if "/login.php" in tree.base_url:
                sys.exit("invalid credentials")
            print("ok", file=STDERR)
        return tree

    @staticmethod
    def _parse_torrent(content: bytes, path: str):
        """decode a torrent, write file list to `path`."""
        info = bdecode(content)[b"info"]
        name = (info[b"name.utf-8"] if b"name.utf-8" in info else
                info[b"name"]).decode(errors="ignore")
        with open(path, "w", encoding="utf-8") as f:
            if b"files" in info:
                files = info[b"files"]
                k = b"path.utf-8" if b"path.utf-8" in files[0] else b"path"
                join = b"/".join
                f.writelines(f'{name}/{join(p[k]).decode(errors="ignore")}\n'
                             for p in files)
            else:
                f.write(name + "\n")


class Analyzer:

    def __init__(self, *, regex_file: str, mteam: dict, **kwargs) -> None:

        self.regex_file = regex_file
        self._reportdir = "report"
        self._mteam = MTeamCollector(**mteam)

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
        except (OSError, ValueError) as e:
            sys.exit(e)
        self.re = re.compile(regex, flags=re.M).search
        self.ext = re.compile(
            r"\.(?:m(?:p4|[24kop]v|2?ts|4p|p2|pe?g|xf)|wmv|"
            r"avi|iso|3gp|asf|bdmv|flv|rm|rmvb|ts|vob|webm)$",
            flags=re.M).search
        os.makedirs(self._reportdir, exist_ok=True)

    def analyze_av(self, local: bool = False):

        print("Matching test begins with av torrents...", file=STDERR)

        reportfile = op.join(self._reportdir, "av_report.txt")
        rawfile = op.join(self._reportdir, "mismatch_raw.txt")
        total = count = 0
        strings = defaultdict(set)
        prefixcount = Counter()
        wordcount = Counter()
        tmp = set()
        prefix_finder = re.compile(
            r"(?:^|/)([0-9]{,5}([a-z]{2,10})"
            r"(?:-[0-9]{2,8}(?:[hm]hb[0-9]{,2}|f?hd|[a-z])?|"
            r"[0-9]{2,8}(?:[hm]hb[0-9]{,2}|f?hd|[a-z]))"
            r")\b.*$",
            flags=re.M).findall
        word_finder = re.compile(r"(?:\b|_)([a-z]{3,})(?:\b|_)", re.M).findall

        paths = self._mteam.from_cache if local else self._mteam.from_web
        paths = paths(is_av=True)

        with ProcessPoolExecutor() as ex, \
            open(rawfile, "w", encoding="utf-8") as f:

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
                    prefixcount.update(tmp)
                    tmp.clear()

                    tmp.update(word_finder(content))
                    wordcount.update(tmp)
                    tmp.clear()
            freqwords = freqwords.result()

        prefixcount = [
            (i, k, strings[k]) for k, i in prefixcount.items() if i >= 5
        ]
        wordcount = [(i, k)
                     for k, i in wordcount.items()
                     if i >= 5 and k not in freqwords]
        f = lambda t: (-t[0], t[1])
        prefixcount.sort(key=f)
        wordcount.sort(key=f)
        stdout_write = sys.stdout.write
        m = max(len("item"), len(f"{wordcount[0][0]}") if wordcount else 0)

        with open(reportfile, "w", encoding="utf-8") as f:
            for line in self._format_report(
                    total=total,
                    count=total - count,
                    title="Potential ID Prefixes",
                    result=prefixcount,
            ):
                stdout_write(line)
                f.write(line)

            f.write("\n\nPotential Keywords:\n"
                    f'{"item":>{m}}  word\n'
                    f'{"-" * 80}\n')
            f.writelines(f"{i:{m}d}  {j}\n" for i, j in wordcount)

        print(f"Report saved to: {op.abspath(reportfile)}", file=STDERR)

    def analyze_nonav(self, local: bool = False):

        print("Matching test begins with non-av torrents...", file=STDERR)

        report_file = op.join(self._reportdir, "nonav_report.txt")
        total = count = 0
        searcher = re.compile(r"[a-z]+").search

        strings = defaultdict(set)
        wordcount = Counter()
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
                    wordcount.update(tmp)
                    tmp.clear()

        wordcount = [(i, k, strings[k]) for k, i in wordcount.items()]
        wordcount.sort(key=lambda t: (-t[0], t[1]))
        stdout_write = sys.stdout.write

        with open(report_file, "w", encoding="utf-8") as f:
            for line in self._format_report(
                    total=total,
                    count=count,
                    title="Matched Strings",
                    result=wordcount,
            ):
                stdout_write(line)
                f.write(line)

        print(f"Report saved to: {op.abspath(report_file)}", file=STDERR)

    def _match_av(self, path: str) -> Optional[str]:
        """If none video is matched, return all videos in the file (in lower
        case)."""
        with open(path, "r", encoding="utf-8") as f:
            a, b = tee(filter(self.ext, map(str.lower, f)))
            if not any(map(self.re, a)):
                return "".join(b)

    def _match_nonav(self, path: str) -> Tuple[str]:
        """Return all matched videos in the file (in lower case). """
        with open(path, "r", encoding="utf-8") as f:
            return tuple(
                m[1]
                for m in map(self.re, filter(self.ext, map(str.lower, f)))
                if m)

    def _format_report(self, total, count, title, result, width=80):

        w1 = max(len("item"), len(f"{result[0][0]}") if result else 0)
        w2 = max(len("word"), max((len(t[1]) for t in result), default=0))
        w3 = max((len(f"{len(t[2])}") for t in result), default=0)
        fmt = f"{{:{w1}d}}  {{:{w2}}}  {{:{w3}d}}: {{}}\n".format
        w4 = width - len(fmt(0, 0, 0, ""))
        slc = self._abbr_slice

        yield (
            f"Regex file: {self.regex_file}\n"
            f"Total: {total:,}, Matched: {count:,}, Percentage: {count / total:.2%}\n\n"
            f"{title}:\n"
            f'{"item":>{w1}}  {"word":{w2}}  strings\n'
            f'{"-" * width}\n')
        for i, k, s in result:
            yield fmt(i, k, len(s), ", ".join(slc(s, w4)))

    @staticmethod
    def _abbr_slice(a: Iterable[str], width: int):
        l = 0
        for x in a:
            l += len(x) + 2
            if l > width:
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
                          status_forcelist=frozenset((500, 502, 503, 504, 521)),
                          backoff_factor=0.3))
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    try:
        with open(path, "rb") as f:
            session.cookies = pickle.load(f)
    except FileNotFoundError:
        os.makedirs("data", exist_ok=True)

    set_cookie = session.cookies.set_cookie
    create_cookie = requests.cookies.create_cookie
    set_cookie(
        create_cookie(domain="www.javbus.com", name="existmag", value="all"))
    set_cookie(
        create_cookie(domain="dmm.co.jp", name="age_check_done", value="1"))
    set_cookie(create_cookie(domain="mgstage.com", name="adc", value="1"))
    set_cookie(
        create_cookie(domain="www.aventertainments.com",
                      name="DVDRowData",
                      value="3"))


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


def progress(iterable, total, start=1, prefix="Progress", width=50):
    """Yield items from iterable while printing a progress bar."""
    if not total:
        return
    write = STDERR.write
    fmt = f"  {prefix}: [{{:{len(str(total))}d}}/{total}] |{{:-<{width}}}| {{:.1%}}\r".format
    for i, obj in enumerate(iterable, start):
        write(fmt(i, "█" * (i * width // total), i / total))
        yield obj
    write("\n")


def dump_cookies(path: str):
    with open(path, "wb") as f:
        pickle.dump(session.cookies, f)


def xp_compile(path: str):
    return XPath(path, regexp=False, smart_strings=False)


def parse_config(configfile: str) -> dict:

    try:
        with open(configfile, "r", encoding="utf-8") as f:
            config = json.load(f)
        norm_configpath(config, "regex_file", "regex.txt")
        norm_configpath(config, "profile_dir", "builder")
        norm_configpath(config["mteam"], "cache_dir", "mteam")
    except FileNotFoundError:
        pass
    except Exception as e:
        sys.exit(f"Config file error: {e}")
    else:
        return config

    default = {
        "regex_file": "regex.txt",
        "keyword_max": 150,
        "prefix_max": 3000,
        "profile_dir": "builder",
        "mteam": {
            "username": "",
            "password": "",
            "page_max": 500,
            "cache_dir": "mteam",
            "av_page": "/adult.php?cat410=1&cat429=1&cat424=1&cat430=1&cat426=1&cat437=1&cat431=1&cat432=1",
            "non_av_page": "/torrents.php",
        },
    } # yapf: disable

    with open(configfile, "w", encoding="utf-8") as f:
        json.dump(default, f, indent=4)
    sys.exit(f"Please edit {configfile} before running me again.")


def norm_configpath(d: dict, k: str, default: str):
    """If a key is missing, or the value is null, set to default value.
    Otherwise normalize the path.
    """
    v = d.get(k)
    d[k] = op.normpath(op.expanduser(v)) if v else default


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
        help="test regex with mteam av torrents",
    )
    group.add_argument(
        "-m",
        "--test-nonav",
        dest="mode",
        action="store_const",
        const="test_nonav",
        help="test regex with mteam non-av torrents",
    )
    group.set_defaults(mode="build")
    parser.add_argument(
        "-l",
        "--local",
        dest="local",
        action="store_true",
        help="use cached data instead of web scraping (default %(default)s)",
    )

    group = parser.add_argument_group(
        title="config override (override corresponding setting in config file)")
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
        help=("maximum mteam pages to scan, override 'mteam.page_max' "
              "(0 for unlimited)"),
    )
    return parser.parse_args()


def main():

    args = parse_arguments()

    path = op.dirname(__file__)
    config = parse_config(op.join(path, "config.json"))
    path = op.join(path, config["profile_dir"])
    try:
        os.chdir(path)
    except FileNotFoundError:
        os.mkdir(path)
        os.chdir(path)
    path = op.join("data", "cookies")
    init_session(path)

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

    dump_cookies(path)


if __name__ == "__main__":
    main()
