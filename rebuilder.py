#!/usr/bin/env python3

import argparse
import pickle
import re
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from configparser import ConfigParser
from itertools import chain, filterfalse, islice, repeat
from operator import itemgetter
from os import chdir
from pathlib import Path
from reprlib import repr as _repr
from time import sleep
from typing import Callable, Iterable, Iterator, List, Optional, Set, Tuple
from urllib.parse import urljoin

from lxml.etree import XPath
from lxml.html import HtmlElement, fromstring
from requests import HTTPError, RequestException, Session
from requests.cookies import create_cookie
from torrentool.api import Torrent
from torrentool.exceptions import TorrentoolException

from regenerator import Regen

_THRESH = 5


class LastPageReached(Exception):
    pass


class JavBusScraper:
    @classmethod
    def get_id(cls):
        id_searcher = re.compile(r"\s*([a-z]{3,8})[_-]?0*([1-9][0-9]{,5})\s*").fullmatch
        for m in filter(None, map(id_searcher, map(str.lower, cls._scrape()))):
            yield m.group(1, 2)

    @staticmethod
    def _scrape() -> Iterator[str]:

        print("Scanning javbus...")
        xp = XPath('.//div[@id="waterfall"]//a[@class="movie-box"]//span/date[1]/text()', smart_strings=False)
        step = 500

        with ThreadPoolExecutor(max_workers=None) as ex:
            for base in ("page", "uncensored/page", "genre/hd", "uncensored/genre/hd"):
                lo = 1
                print(f"  /{base}: ", end="", flush=True)

                while True:
                    hi = lo + step
                    print(f"{lo}:{hi}...", end="", flush=True)
                    urls = (f"https://www.javbus.com/{base}/{i}" for i in range(lo, hi))

                    try:
                        yield from chain.from_iterable(ex.map(_request_map, urls, repeat(xp), repeat(True)))
                    except LastPageReached:
                        break

                    lo = hi

                print()


class JavDBScraper(JavBusScraper):
    @staticmethod
    def _scrape() -> Iterator[str]:

        print(f"Scanning javdb...")
        xp = XPath('.//div[@id="videos"]//a[@class="box"]/div[@class="uid"]/text()', smart_strings=False)

        with ThreadPoolExecutor(max_workers=3) as ex:
            fts = as_completed(
                ex.submit(_request_map, f"https://javdb.com/{p}?page={i}", xp)
                for p in ("uncensored", "")
                for i in range(1, 81)
            )
            yield from chain.from_iterable(map(Future.result, fts))


class GithubScraper(JavBusScraper):
    @staticmethod
    def _scrape() -> List[str]:

        print("Downloading github database...")
        return _request("https://raw.githubusercontent.com/imfht/fanhaodaquan/master/data/codes.json").json()


class MTeamScraper:

    DOMAIN = "https://pt.m-team.cc/"

    def __init__(
        self, av_page: str, non_av_page: str, cache_dir: str, account: Tuple[str, str], limit: int = 500
    ) -> None:

        self._pages = (non_av_page, av_page)
        self._cache_dir = Path(cache_dir)
        self._account = account
        self._range = range(1, limit + 1)
        self._logined = False

    def get_id(self) -> Iterator[Tuple[str, str]]:

        id_searcher = re.compile(
            r"""
            (?:^|/)(?:[0-9]{3})?
            ([a-z]{3,6})
            (-)?
            0*([1-9][0-9]{,3})
            (?(2)(?:hhb[0-9]*)?|hhb[0-9]*)
            \b.*?\.(?:mp4|wmv|avi|mkv|iso)$
            """,
            flags=re.MULTILINE | re.VERBOSE,
        ).search
        freq_words = get_freq_words()

        for path in self.get_path(is_av=True, cjk_title_only=True):
            with open(path, "r", encoding="utf-8") as f:
                for m in filter(None, map(id_searcher, f)):
                    if m[1] not in freq_words:
                        yield m.group(1, 3)

    def get_path(self, is_av: bool, cjk_title_only: bool = False, fetch: bool = True) -> Iterator[Path]:

        subdir = self._cache_dir.joinpath("av" if is_av else "non_av")
        subdir.mkdir(parents=True, exist_ok=True)

        if not fetch:
            matcher = re.compile(r"[0-9]+\.txt").fullmatch
            for path in subdir.iterdir():
                if matcher(path.name):
                    yield path
            return

        pool = []
        page = urljoin(self.DOMAIN, self._pages[is_av])
        matcher = re.compile(r"\bid=([0-9]+)").search

        if cjk_title_only:
            parser = self._get_cjk_parser()
        else:
            parser = XPath(
                './/form[@id = "form_torrent"]'
                '//table[@class = "torrentname"]//a/@href[contains(., "download.php")]',
                smart_strings=False,
            )

        print(f"Scanning mteam... limit: {self._range.stop-1}")
        if not self._logined:
            self._login()

        with ThreadPoolExecutor(max_workers=None) as ex:
            fts = as_completed(ex.submit(_request_map, page, parser, params={"page": i}) for i in self._range)
            for link in chain.from_iterable(map(Future.result, fts)):
                try:
                    path = subdir.joinpath(matcher(link)[1] + ".txt")
                except TypeError:
                    continue
                if path.exists():
                    yield path
                else:
                    pool.append(ex.submit(self._dl_torrent, urljoin(self.DOMAIN, link), path))

            yield from filter(None, map(Future.result, as_completed(pool)))

    def _login(self):
        res = session.head(self.DOMAIN + "torrents.php", allow_redirects=True)
        res.raise_for_status()
        if "/login.php" in res.url:
            res = session.post(
                url=self.DOMAIN + "takelogin.php",
                data={"username": self._account[0], "password": self._account[1]},
                headers={"referer": self.DOMAIN + "login.php"},
            )
            res.raise_for_status()
        self._logined = True

    @staticmethod
    def _get_cjk_parser():
        title_xp = XPath('(.//a[contains(@href, "details.php")]/@title)[1]', smart_strings=False)
        link_xp = XPath('(.//a/@href[contains(.,"download.php")])[1]', smart_strings=False)

        cjk = 0
        for i, j in (
            (4352, 4607),
            (11904, 42191),
            (43072, 43135),
            (44032, 55215),
            (63744, 64255),
            (65072, 65103),
            (65381, 65500),
            (131072, 196607),
        ):
            cjk |= (1 << j + 1) - (1 << i)

        def _parser(tree: HtmlElement):
            for table in tree.iterfind('.//form[@id="form_torrent"]//table[@class="torrentname"]'):
                try:
                    if any(1 << ord(c) & cjk for c in title_xp(table)[0]):
                        yield link_xp(table)[0]
                except IndexError:
                    pass

        return _parser

    @staticmethod
    def _dl_torrent(link: str, path: Path) -> Path:

        print("Downloading:", link)
        try:
            content = _request(link).content
        except (RequestException, AttributeError):
            print(f"Downloading failed: {link}")
            return

        try:
            filelist = Torrent.from_string(content).files
            with open(path, "w", encoding="utf-8") as f:
                f.writelines(i[0].lower() + "\n" for i in filelist)

        except (TorrentoolException, OSError, TypeError):

            torrent_file = path.with_suffix(".torrent")
            spliter = re.compile(r"^\s+(.+) \([^)]+\)$", flags=re.M)

            try:
                torrent_file.write_bytes(content)
                filelist = subprocess.check_output(("transmission-show", torrent_file), text=True)
                filelist = spliter.findall(filelist, filelist.index("\n\nFILES\n\n"))
                if not filelist:
                    raise ValueError

                with open(path, "w", encoding="utf-8") as f:
                    f.writelines(i.lower() + "\n" for i in filelist)

            except (subprocess.CalledProcessError, ValueError, OSError):
                print(f'Parsing torrent error: "{link}"')
                path.unlink(missing_ok=True)
                return

            finally:
                torrent_file.unlink(missing_ok=True)

        return path


class Builder:
    def __init__(self, output_file: str, raw_dir: str, mteam: MTeamScraper, fetch: bool) -> None:

        self._output_file = Path(output_file)
        self._raw_dir = raw_dir = Path(raw_dir)
        self._mteam = mteam
        self._fetch = fetch

        self._kw_file = raw_dir.joinpath("keyword.txt")
        self._prefix_file = raw_dir.joinpath("id_prefix.txt")
        self._whitelist_file = raw_dir.joinpath("id_whitelist.txt")
        self._blacklist_file = raw_dir.joinpath("id_blacklist.txt")

    def build(self):

        self.kw_regex = kw_regex = self._get_regex(
            wordlist=self._update_file(self._kw_file, self._filter_strategy),
            name="Keywords",
            omitOuterParen=True,
        )
        self.prefix_regex = prefix_regex = self._get_regex(
            wordlist=self._update_file(self._prefix_file, self._prefix_strategy),
            name="ID Prefix",
            omitOuterParen=False,
        )
        if not (kw_regex and prefix_regex):
            return

        self.regex = f"(^|[^a-z0-9])({kw_regex}|[0-9]{{,3}}{prefix_regex}[ _-]?[0-9]{{2,8}})([^a-z0-9]|$)"
        return self._update_file(self._output_file, lambda _: (self.regex,))[0]

    @staticmethod
    def _get_regex(wordlist: List[str], name: str, omitOuterParen: bool) -> str:

        regen = Regen(wordlist)
        computed = regen.to_regex(omitOuterParen=omitOuterParen)

        concat = "|".join(wordlist)
        if not omitOuterParen and len(wordlist) > 1:
            concat = f"({concat})"

        diff = len(computed) - len(concat)
        if diff > 0:
            print(f"{name}: Computed regex is {diff} characters longer than concatenation, use the latter.")
            return concat

        regen.verify_result()
        print(f"{name}: Regex test passed. Characters saved: {-diff}.")
        return computed

    @staticmethod
    def _update_file(file: Path, stragety: Callable[[Iterable[str]], Iterable[str]]) -> List[str]:

        try:
            with file.open("r+", encoding="utf-8") as f:
                old_list = f.read().splitlines()
                result = sorted(stragety(old_list))
                if old_list != result:
                    f.seek(0)
                    f.writelines(i + "\n" for i in result)
                    f.truncate()
                    print(f"{file} updated.")

        except FileNotFoundError:
            result = sorted(stragety([]))
            file.parent.mkdir(parents=True, exist_ok=True)
            with file.open(mode="w", encoding="utf-8") as f:
                f.writelines(i + "\n" for i in result)
            print(f"{file} created.")

        return result

    def _web_scrape(self) -> Set[str]:

        result = set(self._mteam.get_id())
        for cls in JavBusScraper, JavDBScraper, GithubScraper:
            result.update(cls.get_id())

        uniq_id = len(result)
        prefix_counter = Counter(map(itemgetter(0), result))
        result = {k for k, v in prefix_counter.items() if v >= _THRESH}

        print(f"Uniq ID: {uniq_id}. Uniq prefix: {len(prefix_counter)}. Final: {len(result)}.")
        return result

    def _prefix_strategy(self, old_list: Iterable[str]) -> Iterator[str]:

        if self._fetch or not old_list:
            result = self._web_scrape()
        else:
            result = self._filter_strategy(old_list)

        result.update(self._update_file(self._whitelist_file, self._extract_strategy))
        result.difference_update(self._update_file(self._blacklist_file, self._extract_strategy))

        return filterfalse(re.compile(self.kw_regex).fullmatch, result)

    @staticmethod
    def _filter_strategy(wordlist: Iterable[str]) -> Set[str]:
        return set(map(str.lower, filter(None, map(str.strip, wordlist))))

    @classmethod
    def _extract_strategy(cls, old_list: Iterable[str]) -> Iterator[str]:
        return Regen(cls._filter_strategy(old_list)).to_text()


class Analyzer:
    def __init__(self, regex_file: str, report_dir: str, mteam: MTeamScraper, fetch: bool) -> None:

        self._report_dir = Path(report_dir)
        self._report_dir.mkdir(parents=True, exist_ok=True)

        self._mteam = mteam
        self._fetch = fetch

        regex = Path(regex_file).read_text(encoding="utf-8").strip()
        assert len(regex.splitlines()) == 1, "regex file should contain only one line"

        p = re.fullmatch(r"(.+?)\((.+)", regex)
        self._av_matcher = re.compile(f"{p[1]}(?P<match>{p[2]}", flags=re.M).search

        self._video_filter = re.compile(
            r"\.(?:3gp|asf|avi|bdmv|flv|iso|m(?:2?ts|4p|[24kop]v|p2|p4|pe?g|xf)|rm|rmvb|ts|vob|webm|wmv)$",
            flags=re.MULTILINE,
        ).search

    def analyze_av(self):

        print("Matching test begins...")

        unmatch_raw = self._report_dir.joinpath("unmatch_raw.txt")
        unmatch_freq = self._report_dir.joinpath("unmatch_frequency.txt")
        total = unmatched = 0
        sep = "-" * 80 + "\n"

        prefix_searcher = re.compile(r"\b[0-9]{,3}([a-z]{2,8})-?[0-9]{2,8}(?:hhb[0-9]*)?\b").search
        word_finder = re.compile(r"(?![\d_]+\b)\w{3,}").findall

        flat_counter = defaultdict(list)
        prefix_counter = Counter()
        word_counter = Counter()
        tmp = set()

        with ProcessPoolExecutor(max_workers=None) as ex, open(unmatch_raw, "w", encoding="utf-8") as f:

            paths = self._mteam.get_path(is_av=True, fetch=self._fetch)

            for video in as_completed(ex.submit(self._match_av, p) for p in paths):

                total += 1
                video = video.result()
                if not video:
                    continue

                unmatched += 1
                f.write(sep)
                f.writelines(video)
                f.write("\n")

                for m in filter(None, map(prefix_searcher, video)):
                    prefix = m[1]
                    flat_counter[prefix].append(m[0])
                    tmp.add(prefix)

                prefix_counter.update(tmp)
                tmp.clear()

                tmp.update(chain.from_iterable(map(word_finder, video)))
                word_counter.update(tmp)
                tmp.clear()

        stat = self._get_stat("Match", total, total - unmatched)
        freq_words = get_freq_words()

        result = [
            (i, len(v), k, set(v))
            for k, v in flat_counter.items()
            if k not in freq_words and (i := prefix_counter[k]) >= _THRESH
        ]
        result.sort(reverse=True)

        words = [(v, k) for k, v in word_counter.items() if k not in freq_words and v >= _THRESH]
        words.sort(reverse=True)

        with open(unmatch_freq, "w", encoding="utf-8") as f:
            f.write(stat + "\n\n")

            f.write("Potential ID Prefixes:\n\n")
            f.writelines(self._format_report(result))

            f.write("\n\nPotential Keywords:\n\n")
            f.write("{:>6}  {}\n{:->80}\n".format("uniq", "word", ""))
            f.writelines(f"{i:6d}  {j}\n" for i, j in words)

        print(stat)
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

            paths = self._mteam.get_path(is_av=False, fetch=self._fetch)

            for ft in as_completed(ex.submit(self._match_non_av, p) for p in paths):

                total += 1
                for m in ft.result():
                    try:
                        word = word_searcher(m)[0]
                    except TypeError:
                        pass
                    else:
                        flat_counter[word].append(m)
                        tmp.add(word)

                if tmp:
                    mismatched += 1
                    torrent_counter.update(tmp)
                    tmp.clear()

        stat = self._get_stat("Mismatch", total, mismatched)
        result = [(torrent_counter[k], len(v), k, set(v)) for k, v in flat_counter.items()]
        result.sort(reverse=True)

        with open(mismatched_file, "w", encoding="utf-8") as f:
            f.write(stat + "\n\n")
            f.writelines(self._format_report(result))

        print(stat)
        print(f"Result saved to: {mismatched_file}")

    def _match_av(self, path: Path) -> Optional[List[str]]:
        result = []
        matcher = self._av_matcher
        with open(path, "r", encoding="utf-8") as f:
            for v in filter(self._video_filter, f):
                if matcher(v):
                    return
                result.append(v)
        return result

    def _match_non_av(self, path: Path) -> Tuple[str]:
        with open(path, "r", encoding="utf-8") as f:
            return tuple(m["match"] for m in map(self._av_matcher, filter(self._video_filter, f)) if m)

    @staticmethod
    def _get_stat(name: str, total: int, n: int):
        return f"Total: {total}. {name}: {n}. Percentage: {n / total * 100:.2f}%."

    @staticmethod
    def _format_report(result: Iterable):
        yield "{:>6}  {:>6}  {:15}  {}\n{:->80}\n".format("uniq", "occur", "word", "strings", "")
        for i, j, k, s in result:
            yield f"{i:6d}  {j:6d}  {k:15}  {_repr(s)[1:-1]}\n"


def _request(url: str, **kwargs):

    for retry in range(3):
        try:
            res = session.get(url, timeout=(7, 28), **kwargs)
            res.raise_for_status()
        except RequestException:
            if retry == 2:
                raise
            sleep(1)
        else:
            return res


def _request_map(url: str, func: Callable, raise_404: bool = False, **kwargs):

    for retry in range(3):
        try:
            res = session.get(url, timeout=(7, 28), **kwargs)
            res.raise_for_status()
        except RequestException as e:
            if raise_404 and isinstance(e, HTTPError) and e.response.status_code == 404:
                raise LastPageReached
            if retry == 2:
                raise
            sleep(1)
        else:
            return func(fromstring(res.content))


_freq_words = None


def get_freq_words(lb: int = 3, ub: int = 6, n: int = 1000):

    global _freq_words

    if _freq_words is None or len(_freq_words) != n:
        word_list = _request(
            "https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english-usa.txt"
        ).text
        _freq_words = frozenset(
            m[0].lower() for m in islice(re.finditer(f"^[A-Za-z]{{{lb},{ub}}}$", word_list, flags=re.M), n)
        )
        assert len(_freq_words) == n, "fetching frequent words failed"

    return _freq_words


def parse_config(configfile: str):

    parser = ConfigParser()

    if parser.read(configfile):
        return parser["DEFAULT"]

    parser["DEFAULT"] = {
        "output_file": "",
        "cache_dir": "",
        "mteam_username": "",
        "mteam_password": "",
        "mteam_av_page": "adult.php?cat410=1&cat429=1&cat426=1&cat437=1&cat431=1&cat432=1",
        "mteam_non_av_page": "torrents.php",
    }

    with open(configfile, "w", encoding="utf-8") as f:
        parser.write(f)

    print(f"Please check {configfile} before running me again.")
    sys.exit()


def parse_arguments():
    parser = argparse.ArgumentParser(description="build and test regex.")

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-b",
        "--build",
        dest="mode",
        action="store_const",
        const="build",
        help="build and save regex to file (default)",
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

    global session

    try:
        with open(session_file, "rb") as f:
            session = pickle.load(f)

        if not isinstance(session, Session):
            raise ValueError

    except (OSError, pickle.PickleError, ValueError):
        session = Session()
        session.cookies.set_cookie(create_cookie(domain="www.javbus.com", name="existmag", value="all"))
        session.headers.update(
            {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:80.0) Gecko/20100101 Firefox/80.0"}
        )


def _save_session(session_file: Path):

    session_file.parent.mkdir(parents=True, exist_ok=True)
    with open(session_file, "wb") as f:
        pickle.dump(session, f)


def main():

    chdir(Path(__file__).parent)

    config = parse_config("builder.ini")
    args = parse_arguments()

    session_file = Path("raw/cookies")
    _init_session(session_file)

    mteam = MTeamScraper(
        av_page=config["mteam_av_page"],
        non_av_page=config["mteam_non_av_page"],
        cache_dir=config["cache_dir"] or "cache",
        account=(config["mteam_username"], config["mteam_password"]),
        limit=args.limit,
    )

    if args.mode == "build":
        regex = Builder(
            output_file=config["output_file"] or "regex.txt",
            raw_dir="raw",
            mteam=mteam,
            fetch=args.fetch,
        ).build()

        if regex:
            print(f"\nResult ({len(regex)} chars):")
            print(regex)
        else:
            print("Generating regex failed.")

    else:
        analyzer = Analyzer(
            regex_file=config["output_file"],
            report_dir="report",
            mteam=mteam,
            fetch=args.fetch,
        )

        if args.mode == "test_match":
            analyzer.analyze_av()
        else:
            analyzer.analyze_non_av()

    _save_session(session_file)


if __name__ == "__main__":
    main()
