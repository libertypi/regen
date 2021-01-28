#!/usr/bin/env python3

import argparse
import json
import os
import os.path as op
import re
import reprlib
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import (ProcessPoolExecutor, ThreadPoolExecutor,
                                as_completed)
from itertools import chain, filterfalse, islice, repeat, tee
from operator import itemgetter
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, Union
from urllib.parse import urljoin, urlsplit

import requests
import yaml
from lxml.etree import XPath
from lxml.html import HtmlElement
from lxml.html import fromstring as html_fromstring
from torrentool.api import Torrent
from urllib3 import Retry

from regen import Regen


class LastPageReached(Exception):
    pass


class Scraper:

    def get_tree(self, url: str, page: int) -> HtmlElement:
        raise NotImplementedError

    def _scrape(self, base: str, paths: Tuple[str], xpath: Union[XPath, str],
                step: int) -> Iterator[str]:

        print(f"Scanning {urlsplit(base).netloc}")
        if isinstance(xpath, str):
            xpath = XPath(xpath, smart_strings=False)

        with ThreadPoolExecutor() as ex:
            for path in paths:
                print(f"  {path}: ", end="", flush=True)
                lo = 1
                path = repeat(base + path)

                while True:
                    print(f"{lo}..", end="", flush=True)
                    hi = lo + step
                    result = ex.map(self.get_tree, path, range(lo, hi))
                    try:
                        yield from chain.from_iterable(map(xpath, result))
                    except LastPageReached as e:
                        print(e)
                        break
                    lo = hi


class JavBusScraper(Scraper):

    def __init__(self, western, jav):

        self.western = western
        self.jav = jav
        self.xpath = XPath(
            './/div[@id="waterfall"]//a[@class="movie-box"]'
            '//span/date[1]/text()',
            smart_strings=False)

    @staticmethod
    def get_tree(url: str, page: int):

        try:
            return html_fromstring(_request(f"{url}/{page}").content)
        except requests.HTTPError as e:
            if e.response.status_code == 404:
                raise LastPageReached(page)
            raise

    def get_keyword(self):

        config = self.western
        result = self._scrape(
            base=config["domain"],
            paths=config["page"],
            xpath=self.xpath,
            step=500,
        )
        matcher = re.compile(r"\s*([A-Za-z0-9]{3,})(?:\.\d\d){3}\s*").fullmatch
        return Counter(m[1].lower() for m in map(matcher, result) if m).items()

    def get_id(self):

        config = self.jav
        return self._scrape(
            base=config["domain"],
            paths=config["page"],
            xpath=self.xpath,
            step=500,
        )


class JavDBScraper(Scraper):

    DOMAIN = "https://javdb.com"

    def __init__(self, western, jav):

        self.western = western
        self.jav = jav
        self.nav_xp = XPath(
            '//section/div[@class="container"]//ul[@class="pagination-list"]/li'
            '/a[contains(@class, "is-current") and normalize-space()=$page]')

    def get_tree(self, url: str, page: int):

        tree = html_fromstring(_request(f"{url}?page={page}").content)
        if self.nav_xp(tree, page=page):
            return tree
        raise LastPageReached(page)

    def get_keyword(self) -> Iterable[Tuple[str, int]]:

        config = self.western
        result = self._scrape(
            base=config["domain"],
            paths=config["page"],
            xpath=('//div[@id="series"]//div[@class="box"]'
                   '/a[@title and strong and span]'),
            step=100,
        )
        matcher = re.compile(r"[a-z0-9]{3,}").fullmatch
        searcher = re.compile(r"\d+").search

        for t in result:
            title = t.findtext("strong").replace(" ", "").lower()
            if matcher(title):
                try:
                    yield title, int(searcher(t.findtext("span"))[0])
                except TypeError:
                    pass

    def get_id(self) -> Iterator[str]:

        config = self.jav
        return self._scrape(
            base=config["domain"],
            paths=config["page"],
            xpath=('.//div[@id="videos"]//a[@class="box"]'
                   '/div[@class="uid"]/text()'),
            step=100,
        )


class OnlineJsonScraper:

    def __init__(self, config: List[str]) -> None:
        self.config = config

    def get_id(self) -> Iterable[str]:

        print("Downloading online json...")
        for url in self.config:
            yield from _request(url).json()


class LocalJSONLoader:

    def __init__(self, config: List[str]) -> None:
        self.filelist = config

    def get_id(self) -> List[str]:

        print("Reading local json...")
        for path in self.filelist:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (OSError, ValueError) as e:
                print(e, file=sys.stderr)
            else:
                yield from data


class Builder:

    def __init__(self, scrapers: List[Scraper], regex_file: str,
                 keyword_max: int, prefix_max: int,
                 custom: Dict[str, List[str]]) -> None:

        self._scrapers = scrapers
        self._regex_file = Path(regex_file)
        self._datafile = Path("data.yaml")
        self._keyword_max = keyword_max
        self._prefix_max = prefix_max
        self._custom = custom

    def from_web(self):

        data = {
            "keyword": self._fetch_keyword(),
            "prefix": self._fetch_prefix(),
        }
        with open(self._datafile, "w", encoding="utf-8") as f:
            yaml.dump(data, f, sort_keys=False, Dumper=yaml.CDumper)

        return self._build(data)

    def from_cache(self):

        try:
            with open(self._datafile, "r", encoding="utf-8") as f:
                data = yaml.load(f, Loader=yaml.CLoader)
        except (OSError, ValueError) as e:
            sys.exit(e)

        return self._build(data)

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

        print("-" * 50)
        if not (keyword and prefix):
            return

        regex = rf"(^|[^a-z0-9])({keyword}|[0-9]{{,3}}{prefix}[_-]?[0-9]{{2,8}})([^a-z0-9]|$)"
        self._update_file(self._regex_file, regex)
        return regex

    def _build_regex(self,
                     name: str,
                     data: Dict[str, dict],
                     omitOuterParen: bool,
                     filterlist: Tuple[str] = ()) -> str:

        print(f" {name.upper()} ".center(50, "-"))
        data = data[name]
        print(f"Entry: {sum(data.values())}, {name}: {len(data)}")

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

        print("Cut: {} (frequency: {})".format(
            len(words), data[words[-1]] if words else None))

        whitelist = self._custom[name + "_whitelist"]
        blacklist = self._custom[name + "_blacklist"]
        regex = chain(whitelist, blacklist, filterlist)
        regex = re.compile("|".join(regex)).fullmatch
        words[:] = filterfalse(regex, words)
        words.extend(whitelist)
        words.sort()

        print(f"Final: {len(words)}")

        regen = Regen(words)
        regex = regen.to_regex(omitOuterParen=omitOuterParen)

        concat = "|".join(words)
        if not omitOuterParen and len(words) > 1:
            concat = f"({concat})"

        length = len(regex)
        diff = length - len(concat)
        if diff > 0:
            print(f"Computed regex is {diff} characters "
                  "longer than concatenation, use the latter.")
            regex = concat
        else:
            regen.raise_for_verify()
            print(f"Regex length: {length} ({diff})")

        return regex

    def _fetch_keyword(self) -> Dict[str, int]:

        d = {}
        setdefault = d.setdefault
        freq = get_freq_words()

        for s in self._scrapers:
            if not hasattr(s, "get_keyword"):
                continue
            for k, v in s.get_keyword():
                if k not in freq and setdefault(k, v) < v:
                    d[k] = v

        return self._sort_dict_by_val(d)

    def _fetch_prefix(self) -> Dict[str, int]:

        d = defaultdict(set)
        r = re.compile(r"\s*\d*([a-z]{3,8})[_-]?(\d{2,8})\s*").fullmatch

        for s in self._scrapers:
            for m in filter(None, map(r, map(str.lower, s.get_id()))):
                d[m[1]].add(int(m[2]))

        return self._sort_dict_by_val(zip(d, map(len, d.values())))

    @staticmethod
    def _sort_dict_by_val(d: Union[Dict, Iterable[Tuple]]):
        """Sort dictionary or 2-tuple iterable by key, largest first."""

        d = sorted(d.items() if isinstance(d, dict) else d)
        d.sort(key=itemgetter(1), reverse=True)
        return dict(d)

    @staticmethod
    def _update_file(file: Path, content: str):

        if not content.endswith("\n"):
            content += "\n"
        try:
            with open(file, "r+", encoding="utf-8") as f:
                old = f.read()
                if old != content:
                    f.seek(0)
                    f.write(content)
                    f.truncate()
                    print(f"Update: {file}")
        except FileNotFoundError:
            file.parent.mkdir(parents=True, exist_ok=True)
            with open(file, mode="w", encoding="utf-8") as f:
                f.write(content)
            print(f"Create: {file}")


class MTeamCollector:

    DOMAIN = "https://pt.m-team.cc"

    def __init__(self, *, username: str, password: str, page_max: int,
                 cache_dir: str, av_page: str, non_av_page: str) -> None:

        cache_dir = op.normpath(cache_dir)
        self._cache_dirs = (
            op.join(cache_dir, "non_av"),
            op.join(cache_dir, "av"),
        )
        self._pages = (
            urljoin(self.DOMAIN, non_av_page),
            urljoin(self.DOMAIN, av_page),
        )
        self._page_max = page_max
        self._account = {"username": username, "password": password}
        self._logined = False
        self._has_transmission = None

        for cache_dir in self._cache_dirs:
            os.makedirs(cache_dir, exist_ok=True)

    def from_cache(self, is_av: bool) -> Iterator[str]:

        matcher = re.compile(r"[0-9]+\.txt").fullmatch
        with os.scandir(self._cache_dirs[is_av]) as it:
            for entry in it:
                if matcher(entry.name):
                    yield entry.path

    def from_web(self, is_av: bool) -> Iterator[str]:

        url = self._pages[is_av]
        cache = self._cache_dirs[is_av]

        matcher = re.compile(r"\bid=([0-9]+)").search
        xpath = XPath(
            '//form[@id="form_torrent"]//table[@class="torrentname"]'
            '/descendant::a[contains(@href, "download.php?")][1]/@href',
            smart_strings=False)

        join = op.join
        exists = op.exists
        pool = {}
        idx = 0
        total = self._page_max
        step = frozenset(range(1, total, total // 10))

        print(f"Scanning mteam ({total} pages)...", end="", flush=True)
        self._login()

        with ThreadPoolExecutor() as ex:

            for ft in as_completed(
                    ex.submit(_request, url, params={"page": i})
                    for i in range(1, total + 1)):

                idx += 1
                if idx in step:
                    print(f"{idx}..", end="", flush=True)

                for link in xpath(html_fromstring(ft.result().content)):
                    try:
                        path = join(cache, matcher(link)[1] + ".txt")
                    except TypeError:
                        continue
                    if exists(path):
                        yield path
                    else:
                        url = urljoin(self.DOMAIN, link)
                        pool[ex.submit(_request, url)] = url, path
            print(total)

            for ft in as_completed(pool):
                url, path = pool[ft]
                print(f"Downloading: {url}")
                try:
                    self._parse_torrent(ft.result().content, path)
                except requests.RequestException as e:
                    print(e, file=sys.stderr)
                except Exception as e:
                    print(e, file=sys.stderr)
                    try:
                        os.unlink(path)
                    except OSError:
                        pass
                else:
                    yield path

    def _login(self):

        if self._logined:
            return
        try:
            res = session.post(
                url=self.DOMAIN + "/takelogin.php",
                data=self._account,
                headers={"referer": self.DOMAIN + "/login.php"},
            )
            res.raise_for_status()
            res = session.head(self._pages[0], allow_redirects=True)
            if "/login.php" in res.url:
                raise requests.RequestException("invalid credentials")
        except requests.RequestException as e:
            sys.exit(f"login failed: {e}")
        self._logined = True

    def _parse_torrent(self, content: bytes, path: str):
        """Parse a torrent, write file list to `path`."""

        try:
            files = Torrent.from_string(content).files
            with open(path, "w", encoding="utf-8") as f:
                f.writelines(i[0].lower() + "\n" for i in files)
        except OSError:
            raise
        except Exception as e:
            if not self._has_transmission:
                if self._has_transmission is False:
                    raise e
                try:
                    subprocess.run("transmission-show", capture_output=True)
                    self._has_transmission = True
                except FileNotFoundError:
                    print(
                        "transmission-show not found. It is recommended to "
                        "install transmission-show to handle more torrents.\n"
                        "In Ubuntu, try: 'sudo apt install transmission-cli'",
                        file=sys.stderr)
                    self._has_transmission = False
                    raise e

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


class Analyzer:

    def __init__(self, *, regex_file: str, mteam: MTeamCollector) -> None:

        try:
            with open(regex_file, "r", encoding="utf-8") as f:
                regex = f.readline().strip()
                if f.read():
                    raise ValueError("regex file should contain only one line")

            if "(?" in regex:
                raise ValueError("regex should not have special group")

            i = regex.index("(", 1)
            regex = "{}({}".format(regex[:i].replace("(", "(?:"),
                                   regex[i + 1:].replace("(", "(?:"))
            self._matcher = re.compile(regex, flags=re.M).search
        except (OSError, ValueError) as e:
            sys.exit(e)

        self._filter = re.compile(
            r"\.(?:m(?:p4|[24kop]v|2?ts|4p|p2|pe?g|xf)|wmv|avi|iso|3gp|asf|bdmv|flv|rm|rmvb|ts|vob|webm)$",
            flags=re.M).search

        self._mteam = mteam

    def analyze_av(self, local: bool = False):

        print("Matching test begins...")

        total = unmatched = 0
        report_file = "unmatch_report.txt"
        raw_file = "unmatch_raw.txt"
        sep = "-" * 80 + "\n"
        groups = defaultdict(list)
        prefix_count = Counter()
        word_count = Counter()
        tmp = set()

        prefix_finder = re.compile(
            r"\b([0-9]{,3}([a-z]{2,8})-?[0-9]{2,8}(?:[hm]hb[0-9]{,2})?)\b.*$",
            flags=re.M).findall
        word_finder = re.compile(r"(?![\d_]+\b)\w{3,}").findall

        paths = self._mteam.from_cache if local else self._mteam.from_web
        paths = paths(is_av=True)

        with ProcessPoolExecutor() as ex, open(raw_file, "w",
                                               encoding="utf-8") as f:

            for content in ex.map(self._match_av, paths, chunksize=100):

                total += 1
                if content:
                    unmatched += 1
                    f.write(sep)
                    f.write(content)

                    for string, prefix in prefix_finder(content):
                        groups[prefix].append(string)
                        tmp.add(prefix)

                    prefix_count.update(tmp)
                    tmp.clear()

                    tmp.update(word_finder(content))
                    word_count.update(tmp)
                    tmp.clear()

        freq_words = get_freq_words()
        brief = self._get_brief("Match", total, total - unmatched)
        print(brief)

        prefixes = [(i, len(v), k, set(v))
                    for k, v in groups.items()
                    if (i := prefix_count[k]) >= 5 and k not in freq_words]
        prefixes.sort(key=lambda t: (-t[0], -t[1], t[2]))

        words = [(v, k)
                 for k, v in word_count.items()
                 if v >= 5 and k not in freq_words]
        words.sort(key=lambda t: (-t[0], t[1]))

        with open(report_file, "w", encoding="utf-8") as f:
            f.write(brief)

            f.write("\n\nPotential ID Prefixes:\n")
            f.writelines(self._format_report(prefixes))

            f.write("\n\nPotential Keywords:\n")
            f.write("{:>6}  {}\n{:->80}\n".format("uniq", "word", ""))
            f.writelines(f"{i:6d}  {j}\n" for i, j in words)

        print(f"Result saved to: {report_file}")

    def analyze_non_av(self, local: bool = False):

        print("Mismatching test begins...")

        total = mismatched = 0
        report_file = "mismatch_report.txt"
        word_searcher = re.compile(r"[a-z]+").search

        groups = defaultdict(list)
        word_count = Counter()
        tmp = set()
        paths = self._mteam.from_cache if local else self._mteam.from_web
        paths = paths(is_av=False)

        with ProcessPoolExecutor() as ex:

            for video in ex.map(self._match_nonav, paths, chunksize=100):

                total += 1
                if video:
                    mismatched += 1
                    for string in video:
                        try:
                            word = word_searcher(string)[0]
                        except TypeError:
                            word = string
                        groups[word].append(string)
                        tmp.add(word)

                    word_count.update(tmp)
                    tmp.clear()

        brief = self._get_brief("Mismatch", total, mismatched)
        print(brief)

        result = [(word_count[k], len(v), k, set(v)) for k, v in groups.items()]
        result.sort(key=lambda t: (-t[0], -t[1], t[2]))

        with open(report_file, "w", encoding="utf-8") as f:
            f.write(brief)
            f.write("\n\n")
            f.writelines(self._format_report(result))

        print(f"Result saved to: {report_file}")

    def _match_av(self, path: str) -> Optional[str]:

        with open(path, "r", encoding="utf-8") as f:
            a, b = tee(filter(self._filter, f))
            if not any(map(self._matcher, a)):
                return "".join(b)

    def _match_nonav(self, path: str) -> Tuple[str]:

        with open(path, "r", encoding="utf-8") as f:
            return tuple(
                m[1] for m in map(self._matcher, filter(self._filter, f)) if m)

    @staticmethod
    def _get_brief(name: str, total: int, n: int):
        return f"Total: {total}. {name}: {n}. Percentage: {n / total:.2%}."

    @staticmethod
    def _format_report(result: Iterable):
        r = reprlib.repr
        yield "{:>6}  {:>6}  {:15}  {}\n{:->80}\n".format(
            "item", "occur", "word", "strings", "")
        for i, j, k, s in result:
            yield f"{i:6d}  {j:6d}  {k:15}  {r(s)[1:-1]}\n"


def _init_session():

    session = requests.Session()
    session.cookies.set_cookie(
        requests.cookies.create_cookie(domain="www.javbus.com",
                                       name="existmag",
                                       value="all"))
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
    return session


def _request(url: str, **kwargs):

    response = session.get(url, timeout=(6.1, 30), **kwargs)
    response.raise_for_status()
    return response


def get_freq_words():
    """Get wordlist of the top 3000 English words which is longer than 3
    letters."""

    result = _request(
        "https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english-usa.txt"
    ).text
    result = islice(
        re.finditer(r"^\s*([A-Za-z]{3,})\s*$", result, flags=re.MULTILINE),
        3000)
    return frozenset(map(str.lower, map(itemgetter(1), result)))


def sort_custom_list(a: list):
    return sorted(set(map(str.lower, filter(None, map(str.strip, a)))))


def parse_config(configfile: str) -> dict:

    default = {
        "regex_file": "regex.txt",
        "keyword_max": 200,
        "prefix_max": 3000,
        "local_json": [],
        "online_json": ["https://raw.githubusercontent.com/imfht/fanhaodaquan/master/data/codes.json"],
        "javbus": {
            "western": {
                "domain": "https://www.javbus.org",
                "page": ["/page", "/studio/1", "/studio/2", "/studio/3", "/studio/4"],
            },
            "jav": {
                "domain": "https://www.javbus.com",
                "page": ["/page", "/uncensored/page", "/genre/hd", "/uncensored/genre/hd"],
            },
        },
        "javdb": {
            "western": {"domain": "https://javdb.com", "page": ["/series/western"]},
            "jav": {"domain": "https://javdb.com", "page": ["/uncensored", "/"]},
        },
        "mteam": {
            "username": "",
            "password": "",
            "page_max": 500,
            "cache_dir": "cache",
            "av_page": "/adult.php?cat410=1&cat429=1&cat424=1&cat430=1&cat426=1&cat437=1&cat431=1&cat432=1",
            "non_av_page": "/torrents.php",
        },
        "custom": {
            "keyword_whitelist": [],
            "keyword_blacklist": [],
            "prefix_whitelist": [],
            "prefix_blacklist": [],
        }
    } # yapf: disable

    try:
        with open(configfile, "r+", encoding="utf-8") as f:
            config = yaml.load(f, Loader=yaml.CLoader)
            old = config["custom"]
            new = {k: sort_custom_list(old[k]) for k in default["custom"]}
            if old != new:
                config["custom"] = new
                f.seek(0)
                yaml.dump(config, f, sort_keys=False, Dumper=yaml.CDumper)
                f.truncate()

        f = op.expanduser
        config["regex_file"] = f(config["regex_file"])
        config["local_json"][:] = map(f, config["local_json"])
        config["mteam"]["cache_dir"] = f(config["mteam"]["cache_dir"])
        return config
    except FileNotFoundError:
        with open(configfile, "w", encoding="utf-8") as f:
            yaml.dump(default, f, sort_keys=False, Dumper=yaml.CDumper)
        sys.exit(f"Please edit {configfile} before running me again.")
    except (KeyError, ValueError) as e:
        sys.exit(f"error in configfile: {e}")


def parse_arguments():
    parser = argparse.ArgumentParser(description="The ultimate Regex builder.")

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
        help="mis-matching test with non-av torrents",
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
        help="maximum mteam pages to scan, override config '[mteam]page_max'",
    )

    return parser.parse_args()


def main():

    config = op.join(op.dirname(__file__), "builder")
    try:
        os.chdir(config)
    except FileNotFoundError:
        os.mkdir(config)
        os.chdir(config)

    config = parse_config("config.yaml")
    args = parse_arguments()

    if args.file:
        config["regex_file"] = args.file

    if args.mode == "build":

        if args.keyword_max is not None:
            config["keyword_max"] = args.keyword_max
        if args.prefix_max is not None:
            config["prefix_max"] = args.prefix_max

        scrapers = (
            JavBusScraper(**config["javbus"]),
            JavDBScraper(**config["javdb"]),
            OnlineJsonScraper(config["online_json"]),
            LocalJSONLoader(config["local_json"]),
        )
        builder = Builder(
            scrapers=scrapers,
            regex_file=config["regex_file"],
            keyword_max=config["keyword_max"],
            prefix_max=config["prefix_max"],
            custom=config["custom"],
        )
        if args.local:
            regex = builder.from_cache()
        else:
            regex = builder.from_web()
        if regex:
            print(f"\nResult ({len(regex)} chars):")
            print(regex)
        else:
            print("Generating regex failed.", file=sys.stderr)

    else:

        if args.mteam_max is not None:
            config["mteam"]["page_max"] = args.mteam_max

        analyzer = Analyzer(
            regex_file=config["regex_file"],
            mteam=MTeamCollector(**config["mteam"]),
        )

        if args.mode == "test_match":
            analyzer.analyze_av(args.local)
        else:
            analyzer.analyze_non_av(args.local)


session = _init_session()

if __name__ == "__main__":
    main()
