#!/usr/bin/env python3

import argparse
import os
import os.path as op
import pickle
import re
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import (ProcessPoolExecutor, ThreadPoolExecutor,
                                as_completed)
from configparser import ConfigParser
from itertools import chain, filterfalse, islice, tee
from operator import itemgetter, methodcaller
from pathlib import Path
from reprlib import repr as _repr
from typing import Callable, Iterable, Iterator, List, Optional, Set, Tuple
from urllib.parse import urljoin

from lxml.etree import XPath
from lxml.html import fromstring as html_fromstring
from requests import HTTPError, RequestException, Session
from requests.adapters import HTTPAdapter
from requests.cookies import create_cookie
from torrentool.api import Torrent
from torrentool.exceptions import TorrentoolException

from regen import Regen

_PREFIX_THRESH = 5
_KEYWORD_THRESH = 10


class LastPageReached(Exception):
    pass


class Scraper:

    @classmethod
    def get_id(cls) -> Iterator[Tuple[str, str]]:

        matcher = re.compile(
            r"\s*([a-z]{3,8})[_-]?0*([1-9][0-9]{,5})\s*").fullmatch

        return map(itemgetter(1, 2),
                   filter(None, map(matcher, map(str.lower, cls._scrape()))))

    @staticmethod
    def _scrape() -> Iterator[str]:
        raise NotImplemented

    @staticmethod
    def get_keyword() -> Iterator[str]:
        raise NotImplemented


class JavBusScraper(Scraper):

    xpath = None

    @classmethod
    def _scrape(cls, base: str = None, paths: tuple = None) -> Iterator[str]:

        if not base:
            base = "https://www.javbus.com/"
            paths = ("page", "uncensored/page", "genre/hd",
                     "uncensored/genre/hd")

        if cls.xpath is None:
            cls.xpath = XPath(
                './/div[@id="waterfall"]//a[@class="movie-box"]//span/date[1]/text()',
                smart_strings=False)
        download = _get_downloader(cls.xpath, raise_404=True)

        step = 500
        print(f"Scanning {base}...")
        with ThreadPoolExecutor(max_workers=None) as ex:
            for path in paths:
                lo = 1
                print(f"  /{path}: ", end="", flush=True)

                while True:
                    hi = lo + step
                    print(f"{lo}:{hi}..", end="", flush=True)
                    try:
                        yield from chain.from_iterable(
                            ex.map(download, (f"{base}/{path}/{page}"
                                              for page in range(lo, hi))))
                    except LastPageReached:
                        break
                    lo = hi
                print()

    @classmethod
    def get_keyword(cls):

        matcher = re.compile(r"\s*([A-Za-z0-9]{3,})(?:\.\d\d){3}\s*").fullmatch
        result = cls._scrape(
            base="https://www.javbus.org/",
            paths=("page", *(f"studio/{i}" for i in range(1, 5))),
        )

        result = Counter(m[1].lower() for m in map(matcher, result) if m)
        for k, v in result.items():
            if v > _KEYWORD_THRESH:
                yield k


class JavDBScraper(Scraper):

    @staticmethod
    def _scrape(paths: Tuple[str] = None,
                limit: int = None,
                xpath: Callable = None) -> Iterator[str]:

        if not paths:
            paths = ("uncensored", "")
            limit = 81
            xpath = XPath(
                './/div[@id="videos"]//a[@class="box"]/div[@class="uid"]/text()',
                smart_strings=False)
        download = _get_downloader(xpath)

        print(f"Scanning javdb...")
        with ThreadPoolExecutor(max_workers=3) as ex:
            fts = as_completed(
                ex.submit(download, f"https://javdb.com/{path}?page={page}")
                for page in range(1, limit)
                for path in paths)
            yield from chain.from_iterable(map(methodcaller("result"), fts))

    @classmethod
    def get_keyword(cls):

        xpath = XPath(
            '//div[@id="series"]//div[@class="box"]'
            f'/a[@title and re:match(span/text(), "[0-9]+") > {_KEYWORD_THRESH}]'
            '/strong/text()',
            namespaces={"re": "http://exslt.org/regular-expressions"},
            smart_strings=False,
        )
        matcher = re.compile(r"[a-z0-9]{3,}").fullmatch

        for title in cls._scrape(paths=("series/western",),
                                 limit=5,
                                 xpath=xpath):

            title = title.replace(" ", "").lower()
            if matcher(title):
                yield title


class GithubScraper(Scraper):

    @staticmethod
    def _scrape() -> List[str]:

        print("Downloading github database...")
        return _request(
            "https://raw.githubusercontent.com/imfht/fanhaodaquan/master/data/codes.json"
        ).json()


class MTeamScraper:

    DOMAIN = "https://pt.m-team.cc/"

    def __init__(self, limit: int, av_page: str, non_av_page: str,
                 username: str, password: str, cache_dir: str) -> None:

        cache_dir = op.normpath(cache_dir)
        self._cache_dirs = (
            op.join(cache_dir, "non_av"),
            op.join(cache_dir, "av"),
        )
        self._limit = limit
        self._pages = (non_av_page, av_page)
        self._account = {"username": username, "password": password}
        self._logined = False

    def get_id(self) -> Iterator[Tuple[str, str]]:

        id_finder = re.compile(
            r"""
            (?:^|/)(?:[0-9]{3})?
            ([a-z]{3,6})
            (-)?
            0*([1-9][0-9]{,3})
            (?(2)(?:[hm]hb[0-9]{,2})?|[hm]hb[0-9]{,2})
            \b.*\.(?:mp4|wmv|avi|mkv|iso)$
            """,
            flags=re.MULTILINE | re.VERBOSE,
        ).finditer
        freq_words = get_freq_words()

        for path in self.get_path(is_av=True, cjk_title_only=True):
            with open(path, "r", encoding="utf-8") as f:
                for m in id_finder(f.read()):
                    if m[1] not in freq_words:
                        yield m.group(1, 3)

    def get_path(self,
                 is_av: bool,
                 fetch: bool = True,
                 cjk_title_only: bool = False) -> Iterator[str]:

        cache_dir = self._cache_dirs[is_av]
        os.makedirs(cache_dir, exist_ok=True)

        if fetch:
            return self._from_web(cache_dir, self._pages[is_av], cjk_title_only)

        return self._from_cache(cache_dir)

    @staticmethod
    def _from_cache(cache_dir: str) -> Iterator[str]:

        matcher = re.compile(r"[0-9]+\.txt").fullmatch
        with os.scandir(cache_dir) as it:
            for entry in it:
                if matcher(entry.name):
                    yield entry.path

    def _from_web(self, cache_dir: str, baseurl: str,
                  cjk_title_only: bool) -> Iterator[str]:

        if cjk_title_only:
            xpath = XPath(
                r"""
                //form[@id="form_torrent"]
                //table[@class="torrentname"
                    and descendant::a[contains(@href, "details.php?")]
                    //text()[re:test(., "[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7a3]")]
                ]/descendant::a[contains(@href, "download.php?")][1]/@href
                """,
                namespaces={"re": "http://exslt.org/regular-expressions"},
                smart_strings=False,
            )
        else:
            xpath = XPath(
                '//form[@id="form_torrent"]//table[@class="torrentname"]'
                '/descendant::a[contains(@href, "download.php?")][1]/@href',
                smart_strings=False,
            )
        download = _get_downloader(xpath)

        pool = []
        idx = 0
        baseurl = urljoin(self.DOMAIN, baseurl)
        matcher = re.compile(r"id=([0-9]+)").search
        step = min((os.cpu_count() + 4) * 3, 32 * 3, self._limit)
        join = op.join
        exists = op.exists

        print(f"Scanning mteam ({self._limit} pages)...", end="", flush=True)
        self._login()

        with ThreadPoolExecutor(max_workers=None) as ex:

            for ft in as_completed(
                    ex.submit(download, baseurl, params={"page": i})
                    for i in range(1, self._limit + 1)):

                idx += 1
                if not idx % step:
                    if idx <= self._limit - step:
                        print(f"{idx}..", end="", flush=True)
                    else:
                        print(f"{idx}..{self._limit}")

                for link in ft.result():
                    try:
                        path = join(cache_dir, matcher(link)[1] + ".txt")
                    except TypeError:
                        continue
                    if exists(path):
                        yield path
                    else:
                        pool.append(ex.submit(self._dl_torrent, link, path))
            if pool:
                yield from filter(
                    None, map(methodcaller("result"), as_completed(pool)))

    def _login(self):

        if self._logined:
            return
        try:
            res = session.head(urljoin(self.DOMAIN, self._pages[0]),
                               allow_redirects=True)
            res.raise_for_status()
            if "/login.php" in res.url:
                res = session.post(
                    url=self.DOMAIN + "takelogin.php",
                    data=self._account,
                    headers={"referer": self.DOMAIN + "login.php"},
                )
                res.raise_for_status()
        except RequestException as e:
            print(f'login failed: {e}', file=sys.stderr)
            sys.exit()
        self._logined = True

    @classmethod
    def _dl_torrent(cls, link: str, path: str):

        link = urljoin(cls.DOMAIN, link)
        print("Downloading:", link)

        try:
            content = _request(link).content
        except RequestException:
            print(f"Downloading failed: {link}", file=sys.stderr)
            return

        try:
            filelist = Torrent.from_string(content).files
            with open(path, "w", encoding="utf-8") as f:
                f.writelines(i[0].lower() + "\n" for i in filelist)

        except (TorrentoolException, TypeError, OSError):

            torrent_file = path + ".torrent"
            spliter = re.compile(r"^\s+(.+) \([^)]+\)$", flags=re.M)

            try:
                with open(torrent_file, "wb") as f:
                    f.write(content)

                filelist = subprocess.check_output(
                    ("transmission-show", torrent_file), text=True)
                filelist = spliter.findall(filelist,
                                           filelist.index("\n\nFILES\n\n"))
                if not filelist:
                    raise ValueError

                with open(path, "w", encoding="utf-8") as f:
                    f.writelines(i.lower() + "\n" for i in filelist)

            except (subprocess.CalledProcessError, ValueError, OSError):
                print(f'Parsing torrent error: "{link}"', file=sys.stderr)
                try:
                    os.unlink(path)
                except OSError:
                    pass
                return

            finally:
                try:
                    os.unlink(torrent_file)
                except OSError:
                    pass

        return path


class Builder:

    def __init__(self,
                 output_file: str,
                 mteam: MTeamScraper,
                 fetch: bool,
                 raw_dir: str = "raw") -> None:

        self._output_file = Path(output_file)
        self._raw_dir = Path(raw_dir)
        self._mteam = mteam
        self._fetch = fetch

    def build(self):

        self.keyword_regex = keyword = self._build_regex(
            name="keyword",
            omitOuterParen=True,
        )
        self.prefix_regex = prefix = self._build_regex(
            name="prefix",
            omitOuterParen=False,
        )
        if not (keyword and prefix):
            return

        self.regex = f"(^|[^a-z0-9])({keyword}|[0-9]{{,3}}{prefix}[_-]?[0-9]{{2,8}})([^a-z0-9]|$)"
        return _update_file(self._output_file, lambda _: (self.regex,))[0]

    def _build_regex(self, name: str, omitOuterParen: bool):

        joinpath = self._raw_dir.joinpath
        wordlist = _update_file(
            joinpath(f"{name}.txt"),
            getattr(self, f"_{name}_strategy"),
        )
        whitelist = _update_file(
            joinpath(f"{name}_whitelist.txt"),
            self._normalize_words,
        )
        blacklist = _update_file(
            joinpath(f"{name}_blacklist.txt"),
            self._normalize_words,
        )

        if name != "keyword" and self.keyword_regex:
            regex = chain(whitelist, blacklist, (self.keyword_regex,))
        else:
            regex = chain(whitelist, blacklist)
        wordlist = set(
            filterfalse(re.compile("|".join(regex)).fullmatch, wordlist))

        wordlist.update(whitelist)
        wordlist.difference_update(blacklist)
        wordlist = sorted(wordlist)

        print(f"{name} chosen: {len(wordlist)}")

        regen = Regen(wordlist)
        computed = regen.to_regex(omitOuterParen=omitOuterParen)

        concat = "|".join(wordlist)
        if not omitOuterParen and len(wordlist) > 1:
            concat = f"({concat})"

        diff = len(computed) - len(concat)
        if diff > 0:
            print(
                f"{name}: Computed regex is {diff} characters longer than concatenation, use the latter."
            )
            return concat

        regen.raise_for_verify()
        print(f"{name} regex test passed, {-diff} characters saved.")
        return computed

    def _keyword_strategy(self, old_list: Iterable[str]) -> Set[str]:

        if not self._fetch and old_list:
            return self._normalize_words(old_list)

        result = set(
            chain(JavBusScraper.get_keyword(), JavDBScraper.get_keyword()))

        result.difference_update(get_freq_words())
        return result

    def _prefix_strategy(self, old_list: Iterable[str]) -> Iterable[str]:

        if not self._fetch and old_list:
            return self._normalize_words(old_list)

        result = set(self._mteam.get_id())
        for cls in JavBusScraper, JavDBScraper, GithubScraper:
            result.update(cls.get_id())

        c = len(result)
        result = Counter(map(itemgetter(0), result))
        print(f"Uniq ID: {c}. Uniq prefix: {len(result)}.")

        return (k for k, v in result.items() if v >= _PREFIX_THRESH)

    @staticmethod
    def _normalize_words(wordlist: Iterable[str]) -> Set[str]:
        return set(map(str.lower, filter(None, map(str.strip, wordlist))))


def _update_file(file: Path, stragety: Callable) -> List[str]:

    try:
        with open(file, "r+", encoding="utf-8") as f:
            old_list = f.read().splitlines()
            new_list = sorted(stragety(old_list))
            if old_list != new_list:
                f.seek(0)
                f.writelines(i + "\n" for i in new_list)
                f.truncate()
                print(f"{file} updated.")

    except FileNotFoundError:
        new_list = sorted(stragety([]))
        file.parent.mkdir(parents=True, exist_ok=True)
        with open(file, mode="w", encoding="utf-8") as f:
            f.writelines(i + "\n" for i in new_list)
        print(f"{file} created.")

    return new_list


class Analyzer:

    def __init__(self,
                 regex_file: str,
                 mteam: MTeamScraper,
                 fetch: bool,
                 report_dir: str = "report") -> None:

        self._mteam = mteam
        self._fetch = fetch
        self._report_dir = Path(report_dir)
        self._report_dir.mkdir(parents=True, exist_ok=True)

        try:
            with open(regex_file, "r", encoding="utf-8") as f:
                regex = f.readline().strip()
                i = regex.index("(", 1)
                if f.read():
                    raise ValueError

        except FileNotFoundError:
            print(f'{regex_file} not found.', file=sys.stderr)
            sys.exit()

        except ValueError:
            print("regex file should contain one and only one line",
                  file=sys.stderr)
            sys.exit()

        self._matcher = re.compile(f"{regex[:i]}(?P<m>{regex[i+1:]}",
                                   flags=re.M).search
        self._filter = re.compile(
            r"\.(?:m(?:p4|[24kop]v|2?ts|4p|p2|pe?g|xf)|wmv|avi|iso|3gp|asf|bdmv|flv|rm|rmvb|ts|vob|webm)$",
            flags=re.M).search

    def analyze_av(self):

        print("Matching test begins...")

        unmatch_raw = self._report_dir.joinpath("unmatch_raw.txt")
        unmatch_freq = self._report_dir.joinpath("unmatch_frequency.txt")
        total = unmatched = 0
        sep = "-" * 80 + "\n"

        prefix_finder = re.compile(
            r"\b([0-9]{,3}([a-z]{2,8})-?[0-9]{2,8}(?:[hm]hb[0-9]{,2})?)\b.*$",
            flags=re.M).findall
        word_finder = re.compile(r"(?![\d_]+\b)\w{3,}").findall

        flat_counter = defaultdict(list)
        prefix_counter = Counter()
        word_counter = Counter()
        tmp = set()

        with ProcessPoolExecutor(max_workers=None) as ex, open(
                unmatch_raw, "w", encoding="utf-8") as f:

            for content in ex.map(
                    self._match_av,
                    self._mteam.get_path(is_av=True, fetch=self._fetch),
                    chunksize=100,
            ):

                total += 1
                if content:
                    unmatched += 1
                    f.write(sep)
                    f.write(content)

                    for string, prefix in prefix_finder(content):
                        flat_counter[prefix].append(string)
                        tmp.add(prefix)

                    prefix_counter.update(tmp)
                    tmp.clear()

                    tmp.update(word_finder(content))
                    word_counter.update(tmp)
                    tmp.clear()

        brief = self._get_brief("Match", total, total - unmatched)
        print(brief)

        freq_words = get_freq_words()
        result = [(i, len(v), k, set(v))
                  for k, v in flat_counter.items()
                  if (i := prefix_counter[k]) >= _PREFIX_THRESH and
                  k not in freq_words]
        result.sort(reverse=True)

        words = [(v, k)
                 for k, v in word_counter.items()
                 if v >= _PREFIX_THRESH and k not in freq_words]
        words.sort(reverse=True)

        with open(unmatch_freq, "w", encoding="utf-8") as f:
            f.write(brief)
            f.write("\n\n")

            f.write("Potential ID Prefixes:\n\n")
            f.writelines(self._format_report(result))

            f.write("\n\nPotential Keywords:\n\n")
            f.write("{:>6}  {}\n{:->80}\n".format("uniq", "word", ""))
            f.writelines(f"{i:6d}  {j}\n" for i, j in words)

        print(f"Result saved to: {unmatch_freq}")

    def analyze_non_av(self):

        print("Mismatch testing begins...")

        mismatched_file = self._report_dir.joinpath("mismatch_frequency.txt")
        word_searcher = re.compile(r"[a-z]+").search
        total = mismatched = 0

        flat_counter = defaultdict(list)
        torrent_counter = Counter()
        tmp = set()

        with ProcessPoolExecutor(max_workers=None) as ex:

            for video in ex.map(
                    self._match_non_av,
                    self._mteam.get_path(is_av=False, fetch=self._fetch),
                    chunksize=100,
            ):

                total += 1
                if video:
                    mismatched += 1
                    for string in video:
                        try:
                            word = word_searcher(string)[0]
                        except TypeError:
                            word = string
                        flat_counter[word].append(string)
                        tmp.add(word)

                    torrent_counter.update(tmp)
                    tmp.clear()

        brief = self._get_brief("Mismatch", total, mismatched)
        print(brief)

        result = [(torrent_counter[k], len(v), k, set(v))
                  for k, v in flat_counter.items()]
        result.sort(reverse=True)

        with open(mismatched_file, "w", encoding="utf-8") as f:
            f.write(brief)
            f.write("\n\n")
            f.writelines(self._format_report(result))

        print(f"Result saved to: {mismatched_file}")

    def _match_av(self, path: str) -> Optional[str]:

        with open(path, "r", encoding="utf-8") as f:
            a, b = tee(filter(self._filter, f))
            if not any(map(self._matcher, a)):
                return "".join(b)

    def _match_non_av(self, path: str) -> Tuple[str]:

        with open(path, "r", encoding="utf-8") as f:
            return tuple(
                match["m"]
                for match in map(self._matcher, filter(self._filter, f))
                if match)

    @staticmethod
    def _get_brief(name: str, total: int, n: int):
        return f"Total: {total}. {name}: {n}. Percentage: {n / total:.2%}."

    @staticmethod
    def _format_report(result: Iterable):
        yield "{:>6}  {:>6}  {:15}  {}\n{:->80}\n".format(
            "uniq", "occur", "word", "strings", "")
        for i, j, k, s in result:
            yield f"{i:6d}  {j:6d}  {k:15}  {_repr(s)[1:-1]}\n"


def _request(url: str, **kwargs):

    response = session.get(url, timeout=(7, 28), **kwargs)
    response.raise_for_status()
    return response


def _get_downloader(xpath: XPath, raise_404: bool = False):

    def downloader(url: str, **kwargs):

        response = session.get(url, timeout=(7, 28), **kwargs)
        try:
            response.raise_for_status()
        except HTTPError:
            if raise_404 and response.status_code == 404:
                raise LastPageReached
            raise
        return xpath(html_fromstring(response.content))

    return downloader


_freq_words = None


def get_freq_words():
    """Get wordlist of the top 3000 English words which longer than 3 letters."""

    global _freq_words
    if not _freq_words:
        result = _request(
            "https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english-usa.txt"
        ).text
        result = islice(
            re.finditer(r"^\s*([A-Za-z]{3,})\s*$", result, flags=re.M), 3000)
        _freq_words = frozenset(map(str.lower, map(itemgetter(1), result)))
    return _freq_words


def parse_config(configfile: str):

    parser = ConfigParser()
    parser["DEFAULT"] = {
        "output_file":
            "regex.txt",
        "cache_dir":
            "cache",
        "mteam_username":
            "",
        "mteam_password":
            "",
        "mteam_av_page":
            "adult.php?cat410=1&cat429=1&cat424=1&cat430=1&cat426=1&cat437=1&cat431=1&cat432=1",
        "mteam_non_av_page":
            "torrents.php",
    }

    if parser.read(configfile):
        config = parser["DEFAULT"]
        config["output_file"] = op.expanduser(config["output_file"])
        config["cache_dir"] = op.expanduser(config["cache_dir"])
        return config

    with open(configfile, "w", encoding="utf-8") as f:
        parser.write(f)

    print(f"Please check {configfile} before running me again.")
    sys.exit()


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
        help="test regex with av torrents",
    )
    group.add_argument(
        "-m",
        "--test-miss",
        dest="mode",
        action="store_const",
        const="test_miss",
        help="test regex with non-av torrents",
    )
    group.set_defaults(mode="build")

    parser.add_argument(
        "-f",
        "--fetch",
        dest="fetch",
        action="store_true",
        help="fetch info from web instead of using local cache",
    )
    parser.add_argument(
        "-l",
        dest="limit",
        action="store",
        type=int,
        help="mteam page limit (default: %(default)s)",
        default=500,
    )

    args = parser.parse_args()

    if args.limit <= 0:
        parser.error("limit should be an integer greater than zero")

    return args


def _init_session(session_file: Path):

    try:
        with open(session_file, "rb") as f:
            session = pickle.load(f)
    except (OSError, pickle.PickleError):
        pass
    else:
        if isinstance(session, Session):
            return session

    session = Session()
    session.cookies.set_cookie(
        create_cookie(domain="www.javbus.com", name="existmag", value="all"))
    session.headers.update({
        "User-Agent":
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:80.0) Gecko/20100101 Firefox/80.0"
    })
    adapter = HTTPAdapter(max_retries=5)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def _save_session(session: Session, session_file: Path):

    try:
        with open(session_file, "wb") as f:
            pickle.dump(session, f)
    except FileNotFoundError:
        try:
            session_file.parent.mkdir(parents=True)
        except OSError:
            pass
        else:
            _save_session(session, session_file)


session = None


def main():

    global session

    os.chdir(Path(__file__).parent)

    config = parse_config("builder.ini")
    args = parse_arguments()

    session_file = Path("raw/cookies")
    session = _init_session(session_file)

    mteam = MTeamScraper(
        limit=args.limit,
        av_page=config["mteam_av_page"],
        non_av_page=config["mteam_non_av_page"],
        username=config["mteam_username"],
        password=config["mteam_password"],
        cache_dir=config["cache_dir"],
    )

    if args.mode == "build":
        regex = Builder(
            output_file=config["output_file"],
            mteam=mteam,
            fetch=args.fetch,
        ).build()

        if regex:
            print(f"\nResult ({len(regex)} chars):")
            print(regex)
        else:
            print("Generating regex failed.", file=sys.stderr)

    else:
        analyzer = Analyzer(
            regex_file=config["output_file"],
            mteam=mteam,
            fetch=args.fetch,
        )

        if args.mode == "test_match":
            analyzer.analyze_av()
        else:
            analyzer.analyze_non_av()

    _save_session(session, session_file)


if __name__ == "__main__":
    main()
