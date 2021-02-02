#!/usr/bin/env python3

import argparse
import json
import os
import os.path as op
import pickle
import re
import reprlib
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import (ProcessPoolExecutor, ThreadPoolExecutor,
                                as_completed)
from itertools import chain, filterfalse, islice, repeat, tee
from operator import itemgetter
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, Union
from urllib.parse import urljoin, urlsplit

import requests
from lxml.etree import XPath
from lxml.html import HtmlElement
from lxml.html import fromstring as html_fromstring
from torrentool.api import Torrent
from urllib3 import Retry

from regen import Regen

STDERR = sys.stderr
session = None


class LastPageReached(Exception):
    pass


class Scraper:

    __slots__ = ("ex", "western", "jav")
    ex: ThreadPoolExecutor
    STEP: int = 500

    def get_tree(self, url: str, i: int) -> HtmlElement:
        raise NotImplementedError

    def _scrape(self, xpath: Union[XPath, str], domain: str, pages: Tuple[str]):

        print(f"Scanning {urlsplit(domain).netloc}", file=STDERR)
        if isinstance(xpath, str):
            xpath = XPath(xpath, smart_strings=False)

        for page in pages:
            lo = 1
            print(f"  {page}: ", end="", flush=True, file=STDERR)
            url = repeat(urljoin(domain, page))
            try:
                while True:
                    hi = lo + self.STEP
                    print(f"{lo}..", end="", flush=True, file=STDERR)
                    trees = self.ex.map(self.get_tree, url, range(lo, hi))
                    yield from chain.from_iterable(map(xpath, trees))
                    lo = hi
            except LastPageReached as e:
                print(e, file=STDERR)


class JavBusScraper(Scraper):

    __slots__ = "xpath"

    def __init__(self, ex, western, jav):
        self.ex = ex
        self.western = western
        self.jav = jav
        self.xpath = XPath(
            './/div[@id="waterfall"]//a[@class="movie-box"]'
            '//span/date[1]/text()',
            smart_strings=False)

    @staticmethod
    def get_tree(url: str, i: int):
        try:
            return html_fromstring(_request(f"{url}/{i}").content)
        except requests.HTTPError as e:
            if e.response.status_code == 404:
                raise LastPageReached(i)
            raise

    def get_keyword(self):
        f = re.compile(r"\s*([A-Za-z0-9]{3,})(?:\.\d\d){3}\s*").fullmatch
        r = self._scrape(self.xpath, **self.western)
        return Counter(m[1].lower() for m in map(f, r) if m).items()

    def get_id(self):
        return self._scrape(self.xpath, **self.jav)


class JavDBScraper(Scraper):

    __slots__ = "nav_xp"
    STEP = 100

    def __init__(self, ex, western, jav):
        self.ex = ex
        self.western = western
        self.jav = jav
        self.nav_xp = XPath(
            '//section/div[@class="container"]//ul[@class="pagination-list"]/li'
            '/a[contains(@class, "is-current") and normalize-space()=$page]')

    def get_tree(self, url: str, i: int):
        tree = html_fromstring(_request(url, params={"page": i}).content)
        if self.nav_xp(tree, page=i):
            return tree
        raise LastPageReached(i)

    def get_keyword(self) -> Iterable[Tuple[str, int]]:
        matcher = re.compile(r"[a-z0-9]{3,}").fullmatch
        searcher = re.compile(r"\d+").search
        xpath = '//div[@id="series"]//div[@class="box"]/a[@title and strong and span]'

        for a in self._scrape(xpath, **self.western):
            title = a.findtext("strong").replace(" ", "").lower()
            if matcher(title):
                try:
                    yield title, int(searcher(a.findtext("span"))[0])
                except TypeError:
                    pass

    def get_id(self) -> Iterator[str]:
        xpath = './/div[@id="videos"]//a[@class="box"]/div[@class="uid"]/text()'
        return self._scrape(xpath, **self.jav)


class OnlineJsonScraper:

    __slots__ = ("ex", "config")

    def __init__(self, ex, config: List[str]) -> None:
        self.ex = ex
        self.config = config

    def get_id(self) -> Iterable[str]:
        print("Downloading online json...", file=STDERR)
        if len(self.config) == 1:
            return _request(self.config[0]).json()
        return self._get_id_multifiles()

    def _get_id_multifiles(self) -> Iterable[str]:
        for ft in as_completed(
                self.ex.submit(_request, u) for u in self.config):
            yield from ft.result().json()


class LocalJSONLoader:

    __slots__ = "config"

    def __init__(self, config: List[str]) -> None:
        self.config = config

    def get_id(self) -> Iterable[str]:
        print("Reading local json...", file=STDERR)
        if len(self.config) == 1:
            with open(self.config[0], "rb") as f:
                return json.load(f)
        return self._get_id_multifiles()

    def _get_id_multifiles(self) -> Iterable[str]:
        for path in self.config:
            with open(path, "rb") as f:
                data = json.load(f)
            yield from data


class MTeamCollector:

    def __init__(self, *, username: str, password: str, page_max: int,
                 cache_dir: str, domain: str, av_page: str,
                 non_av_page: str) -> None:

        self._account = {"username": username, "password": password}
        self._page_max = page_max
        self._cache_dirs = (
            op.join(cache_dir, "non_av"),
            op.join(cache_dir, "av"),
        )
        self.domain = domain
        self._urls = (urljoin(domain, non_av_page), urljoin(domain, av_page))

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
        join = op.join
        exists = op.exists
        pool = {}
        idx = 0
        total = self._page_max
        step = frozenset(range(1, total, (total // 10) or 1))
        matcher = re.compile(r"\bid=([0-9]+)").search
        xpath = XPath(
            '//form[@id="form_torrent"]//table[@class="torrentname"]'
            '/descendant::a[contains(@href, "download.php?")][1]/@href',
            smart_strings=False)

        print(f"Scanning mteam ({total} pages)...",
              end="",
              flush=True,
              file=STDERR)
        self._login()

        with ThreadPoolExecutor() as ex:

            for ft in as_completed(
                    ex.submit(_request, url, params={"page": i})
                    for i in range(1, total + 1)):

                idx += 1
                if idx in step:
                    print(f"{idx}..", end="", flush=True, file=STDERR)

                for link in xpath(html_fromstring(ft.result().content)):
                    try:
                        path = join(cache_dir, matcher(link)[1] + ".txt")
                    except TypeError:
                        continue
                    if exists(path):
                        yield path
                    else:
                        url = urljoin(self.domain, link)
                        pool[ex.submit(_request, url)] = url, path
            print(total, file=STDERR)

            for ft in as_completed(pool):
                url, path = pool[ft]
                print(f"Downloading: {url}", file=STDERR)
                try:
                    self._parse_torrent(ft.result().content, path)
                except requests.RequestException as e:
                    print(e, file=STDERR)
                except Exception as e:
                    print(f"Error: {e}", file=STDERR)
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
                r = session.post(
                    url=urljoin(self.domain, "/takelogin.php"),
                    data=self._account,
                    headers={"referer": urljoin(self.domain, "/login.php")},
                )
                r.raise_for_status()
            else:
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
            self._parse_torrent2(content, path, e=e)

    def _parse_torrent2(self, content: bytes, path: str, e: Exception):

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
    def _skip(*args, e: Exception = None):
        if e is not None:
            raise e


class Builder:

    def __init__(self, config: dict) -> None:

        self._config = config
        self._regex_file = config["regex_file"]
        self._keyword_max = config["keyword_max"]
        self._prefix_max = config["prefix_max"]
        self._datafile = "data.json"
        self._customfile = "custom.yaml"

    def from_web(self) -> Optional[str]:

        config = self._config
        with ThreadPoolExecutor() as ex:
            scrapers = (
                JavBusScraper(ex, **config["javbus"]),
                JavDBScraper(ex, **config["javdb"]),
                OnlineJsonScraper(ex, config["online_json"]),
                LocalJSONLoader(config["local_json"]),
            )
            data = {
                "keyword": self._scrape_keyword(scrapers),
                "prefix": self._scrape_prefix(scrapers),
            }
        with open(self._datafile, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        return self._build(data)

    def from_cache(self) -> Optional[str]:

        try:
            with open(self._datafile, "rb") as f:
                data = json.load(f)
        except (OSError, ValueError) as e:
            sys.exit(e)
        return self._build(data)

    def _scrape_keyword(self, scrapers) -> Dict[str, int]:

        d = {}
        setdefault = d.setdefault
        freq = get_freq_words()

        for s in scrapers:
            if not hasattr(s, "get_keyword"):
                continue
            for k, v in s.get_keyword():
                if k not in freq and setdefault(k, v) < v:
                    d[k] = v
        return self._sort_scrape(d.items())

    def _scrape_prefix(self, scrapers) -> Dict[str, int]:

        d = defaultdict(set)
        r = re.compile(r"\s*\d*([a-z]{3,8})[_-]?(\d{2,8})\s*").fullmatch

        for s in scrapers:
            for m in filter(None, map(r, map(str.lower, s.get_id()))):
                d[m[1]].add(int(m[2]))
        return self._sort_scrape(zip(d, map(len, d.values())))

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

        regex = rf"(^|[^a-z0-9])({keyword}|[0-9]{{,3}}{prefix}[_-]?[0-9]{{2,8}})([^a-z0-9]|$)"
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

    def _update_file(self, file: str, content: str = None) -> List[str]:

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

    def __init__(self, config: dict) -> None:

        self.regex_file = config["regex_file"]
        try:
            with open(self.regex_file, "r", encoding="utf-8") as f:
                regex = f.readline().strip()
                if f.read():
                    raise ValueError("regex file should contain only one line")

            if "(?" in regex:
                raise ValueError("regex should have no special groups")

            i = regex.index("(", 1)
            regex = "{}({}".format(regex[:i].replace("(", "(?:"),
                                   regex[i + 1:].replace("(", "(?:"))
            self.matcher = re.compile(regex, flags=re.M).search
        except (OSError, ValueError) as e:
            sys.exit(e)

        self.extfilter = re.compile(
            r"\.(?:m(?:p4|[24kop]v|2?ts|4p|p2|pe?g|xf)|wmv|avi|iso|3gp|asf|bdmv|flv|rm|rmvb|ts|vob|webm)$",
            flags=re.M).search
        self._mteam = MTeamCollector(**config["mteam"])

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
            r"\b([0-9]{,3}([a-z]{2,8})-?[0-9]{2,8}(?:[hm]hb[0-9]{,2})?)\b.*$",
            flags=re.M).findall
        word_finder = re.compile(r"(?![\d_]+\b)\w{3,}").findall

        paths = self._mteam.from_cache if local else self._mteam.from_web
        paths = paths(is_av=True)

        with ProcessPoolExecutor() as ex, \
             open(raw_file, "w", encoding="utf-8") as f:

            freq_words = ex.submit(get_freq_words)
            for content in ex.map(self._match_av, paths, chunksize=100):
                total += 1
                if content:
                    count += 1
                    f.write("---\n")
                    f.write(content)
                    for string, prefix in prefix_finder(content):
                        strings[prefix].add(string)
                        tmp.add(prefix)
                    prefix_count.update(tmp)
                    tmp.clear()
                    tmp.update(word_finder(content))
                    word_count.update(tmp)
                    tmp.clear()
            freq_words = freq_words.result()

        prefix_count = [(i, k, strings[k])
                        for k, i in prefix_count.items()
                        if i >= 5 and k not in freq_words]
        word_count = [(i, k)
                      for k, i in word_count.items()
                      if i >= 5 and k not in freq_words]

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
                    f'{"torrent":>7}  word\n{"-" * 80}\n')
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
            a, b = tee(filter(self.extfilter, f))
            if not any(map(self.matcher, a)):
                return "".join(b)

    def _match_nonav(self, path: str) -> Tuple[str]:
        with open(path, "r", encoding="utf-8") as f:
            return tuple(
                m[1] for m in map(self.matcher, filter(self.extfilter, f)) if m)

    def _format_report(self, total, count, title, result):
        r = reprlib.repr
        yield (
            f"Regex file: {self.regex_file}\n"
            f"Total: {total}, Matched: {count}, Percentage: {count / total:.2%}\n\n"
            f"{title}:\n"
            f'{"torrent":>7}  {"word":16}strings\n{"-" * 80}\n')
        for i, k, s in result:
            yield f"{i:7d}  {k:16}{r(s)[1:-1]}\n"


def get_freq_words(k: int = 3000):
    """Get wordlist of the top `k` English words which is longer than 3
    letters."""
    u = "https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english-usa.txt"
    m = re.finditer(r"^\s*([A-Za-z]{3,})\s*$", _request(u).text, re.M)
    return frozenset(map(str.lower, map(itemgetter(1), islice(m, k))))


def _request(url: str, **kwargs):
    response = session.get(url, timeout=(9.1, 30), **kwargs)
    response.raise_for_status()
    return response


def init_session(path: str):
    global session
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:80.0) "
                      "Gecko/20100101 Firefox/80.0"
    })
    adapter = requests.adapters.HTTPAdapter(
        max_retries=Retry(total=5,
                          status_forcelist=frozenset((500, 502, 503, 504)),
                          backoff_factor=0.3))
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    try:
        with open(path, "rb") as f:
            session.cookies = pickle.load(f)
    except FileNotFoundError:
        session.cookies.set_cookie(
            requests.cookies.create_cookie(domain="www.javbus.com",
                                           name="existmag",
                                           value="all"))


def dump_cookies(path: str):
    with open(path, "wb") as f:
        pickle.dump(session.cookies, f)


def parse_config(configfile: str) -> dict:
    try:
        with open(configfile, "rb") as f:
            config = json.load(f)
        a = op.normpath
        b = op.expanduser
        config["regex_file"] = a(b(config["regex_file"]))
        config["local_json"][:] = (a(b(p)) for p in config["local_json"])
        config["mteam"]["cache_dir"] = a(b(config["mteam"]["cache_dir"]))
    except FileNotFoundError:
        pass
    except Exception as e:
        sys.exit(f"Error in config file: {e}")
    else:
        return config

    default = {
        "regex_file": "regex.txt",
        "keyword_max": 200,
        "prefix_max": 3000,
        "local_json": [],
        "online_json": ["https://raw.githubusercontent.com/imfht/fanhaodaquan/master/data/codes.json"],
        "javbus": {
            "western": {
                "domain": "https://www.javbus.org",
                "pages": ["/page", "/studio/1", "/studio/2", "/studio/3", "/studio/4"],
            },
            "jav": {
                "domain": "https://www.javbus.com",
                "pages": ["/page", "/uncensored/page", "/genre/hd", "/uncensored/genre/hd"],
            },
        },
        "javdb": {
            "western": {"domain": "https://javdb.com", "pages": ["/series/western"]},
            "jav": {"domain": "https://javdb.com", "pages": ["/uncensored", "/"]},
        },
        "mteam": {
            "username": "",
            "password": "",
            "page_max": 500,
            "cache_dir": "cache",
            "domain": "https://pt.m-team.cc",
            "av_page": "/adult.php?cat410=1&cat429=1&cat424=1&cat430=1&cat426=1&cat437=1&cat431=1&cat432=1",
            "non_av_page": "/torrents.php",
        },
    } # yapf: disable

    with open(configfile, "w", encoding="utf-8") as f:
        json.dump(default, f, indent=4)
    sys.exit(f"Please edit {configfile} before running me again.")


def parse_arguments():

    parser = argparse.ArgumentParser(
        description="The ultimate Regex builder, by David Pi.")

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-b",
        "--build",
        dest="mode",
        action="store_const",
        const="build",
        help="build regex and save to file (default)",
    )
    group.add_argument(
        "-t",
        "--test-match",
        dest="mode",
        action="store_const",
        const="test_match",
        help="matching test with av torrents",
    )
    group.add_argument(
        "-m",
        "--test-mis",
        dest="mode",
        action="store_const",
        const="test_mis",
        help="mismatching test with non-av torrents",
    )
    group.set_defaults(mode="build")

    parser.add_argument(
        "-l",
        "--local",
        dest="local",
        action="store_true",
        help="use cached data instead of web scraping "
        "(default %(default)s)",
    )
    parser.add_argument(
        "-f",
        "--file",
        dest="file",
        action="store",
        help="the target file, override config 'regex_file'",
    )
    parser.add_argument(
        "--kmax",
        dest="keyword_max",
        action="store",
        type=int,
        help="maximum keywords, override config 'keyword_max' (0 for unlimited)",
    )
    parser.add_argument(
        "--pmax",
        dest="prefix_max",
        action="store",
        type=int,
        help="maximum prefixes, override config 'prefix_max' (0 for unlimited)",
    )
    parser.add_argument(
        "--mtmax",
        dest="mteam_max",
        action="store",
        type=int,
        help="maximum mteam pages to scan, override config 'mteam/page_max'",
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

        builder = Builder(config)
        if args.local:
            builder.from_cache()
        else:
            builder.from_web()

    else:

        if args.mteam_max is not None:
            config["mteam"]["page_max"] = args.mteam_max

        analyzer = Analyzer(config)
        if args.mode == "test_match":
            analyzer.analyze_av(args.local)
        else:
            analyzer.analyze_nonav(args.local)

    dump_cookies("cookies")


if __name__ == "__main__":
    main()
