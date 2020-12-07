#!/usr/bin/env python3

import argparse
import pickle
import re
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from configparser import ConfigParser
from itertools import chain, filterfalse
from operator import itemgetter
from os import chdir
from pathlib import Path
from reprlib import repr as _repr
from time import sleep
from typing import Callable, Iterable, Iterator, List, Optional, Set, Tuple
from urllib.parse import urljoin

import requests
from lxml.etree import XPath
from lxml.html import HtmlElement, fromstring
from requests.cookies import create_cookie
from torrentool.api import Torrent
from torrentool.exceptions import TorrentoolException

from regenerator import Regen

_THRESH = 5


class LastPageReached(Exception):
    pass


class MteamScraper:

    DOMAIN = "https://pt.m-team.cc/"

    def __init__(self, cache_dir: Path, account: Tuple[str, str], fetch: bool = False) -> None:

        self._cache_dir = cache_dir
        self._account = account
        self._fetch = fetch

    def run(self, page: str, lo: int, hi: int, is_av: bool, cjk_only: bool = False) -> Iterator[Path]:

        subdir = self._cache_dir.joinpath("av" if is_av else "non_av")
        subdir.mkdir(parents=True, exist_ok=True)

        if not self._fetch:
            path = None
            matcher = re.compile(r"[0-9]+\.txt").fullmatch
            for path in subdir.iterdir():
                if matcher(path.name):
                    yield path
            if path is not None:
                return

        page = urljoin(self.DOMAIN, page)
        id_searcher = re.compile(r"\bid=([0-9]+)").search

        if cjk_only:
            parser = self._make_cjk_parser()
        else:
            parser = XPath(
                '//form[@id="form_torrent"]//table[@class="torrentname"]//a[contains(@href, "download.php")]/@href'
            )

        self._login()
        pool = []
        with ThreadPoolExecutor(max_workers=None) as ex:
            for ft in as_completed(ex.submit(self._get_link, page, i, parser) for i in range(lo, hi)):
                for link in ft.result():
                    try:
                        path = subdir.joinpath(id_searcher(link).expand(r"\1.txt"))
                    except AttributeError:
                        continue
                    if path.exists():
                        yield path
                    else:
                        pool.append(ex.submit(self._download, urljoin(self.DOMAIN, link), path))
            yield from filter(None, map(Future.result, as_completed(pool)))

    def _login(self):
        for retry in range(3):
            try:
                res = session.head(self.DOMAIN + "torrents.php", allow_redirects=True)
                res.raise_for_status()
                if "/login.php" in res.url:
                    session.post(
                        url=f"{self.DOMAIN}/takelogin.php",
                        data={"username": self._account[0], "password": self._account[1]},
                        headers={"referer": f"{self.DOMAIN}/login.php"},
                    )
            except requests.RequestException:
                if retry == 2:
                    raise
                sleep(1)
            else:
                return

    @staticmethod
    def _make_cjk_parser():
        table_path = './/form[@id="form_torrent"]//table[@class="torrentname"]'
        title_xp = XPath('(.//a[contains(@href, "details.php")]/@title)[1]')
        link_xp = XPath('(.//a[contains(@href, "download.php")]/@href)[1]')

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

        def _parser(tree: HtmlElement) -> List[str]:
            result = []
            for table in tree.iterfind(table_path):
                try:
                    if any(1 << ord(c) & cjk for c in title_xp(table)[0]):
                        result.append(link_xp(table)[0])
                except IndexError:
                    pass
            return result

        return _parser

    @staticmethod
    def _get_link(page: str, n: int, parser: Callable) -> List[str]:

        for retry in range(3):
            try:
                r = session.get(page, timeout=(7, 27), params={"page": n})
                r.raise_for_status()
            except requests.RequestException:
                if retry == 2:
                    raise
                sleep(1)
            else:
                return parser(fromstring(r.content))

    @staticmethod
    def _download(link: str, path: Path) -> Path:

        print("Downloading:", link)

        for _ in range(3):
            try:
                content = session.get(link, timeout=(7, 27)).content
            except (requests.RequestException, AttributeError):
                sleep(1)
            else:
                break
        else:
            print(f"Downloading torrent failed: {link}")
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
                return

            finally:
                torrent_file.unlink(missing_ok=True)

        return path


class JavREBuilder:
    def __init__(
        self,
        output_file: Path,
        raw_dir: Path,
        mteam_scraper: MteamScraper,
        mteam_page: str,
        mteam_limit: int,
        fetch: bool,
    ):

        self.output_file = output_file
        raw_dir.mkdir(parents=True, exist_ok=True)

        self.kw_file = raw_dir.joinpath("keyword.txt")
        self.prefix_file = raw_dir.joinpath("id_prefix.txt")
        self.whitelist_file = raw_dir.joinpath("id_whitelist.txt")
        self.blacklist_file = raw_dir.joinpath("id_blacklist.txt")

        self.mteam_scrape = mteam_scraper.run(mteam_page, 1, mteam_limit, is_av=True, cjk_only=True)
        self._fetch = fetch

    def run(self):

        kw_regex = self._get_regex(
            wordlist=self._update_file(self.kw_file, self._filter_strategy),
            name="Keywords",
            omitOuterParen=True,
        )
        self._kw_filter = re.compile(kw_regex).fullmatch

        prefix_regex = self._get_regex(
            wordlist=self._update_file(self.prefix_file, self._prefix_strategy),
            name="ID Prefix",
            omitOuterParen=False,
        )

        if not (kw_regex and prefix_regex):
            print("Generating regex failed.")
            return

        self.regex = f"(^|[^a-z0-9])({kw_regex}|[0-9]{{,3}}{prefix_regex}[ _-]?[0-9]{{2,6}})([^a-z0-9]|$)"
        return self._update_file(self.output_file, lambda _: (self.regex,))[0]

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
            with file.open(mode="w", encoding="utf-8") as f:
                f.writelines(i + "\n" for i in result)
            print(f"{file} created.")

        return result

    def _prefix_strategy(self, old_list: Iterable[str]) -> Iterator[str]:

        if self._fetch or not old_list:
            result = self._web_scrape()
        else:
            result = self._filter_strategy(old_list)

        result.update(self._update_file(self.whitelist_file, self._extract_strategy))
        result.difference_update(self._update_file(self.blacklist_file, self._extract_strategy))

        return filterfalse(self._kw_filter, result)

    @classmethod
    def _extract_strategy(cls, old_list: Iterable[str]) -> Iterator[str]:
        return Regen(cls._filter_strategy(old_list)).to_text()

    @staticmethod
    def _filter_strategy(wordlist: Iterable[str]) -> Set[str]:
        return set(map(str.lower, filter(None, map(str.strip, wordlist))))

    def _web_scrape(self) -> Set[str]:

        result = set(self._scrape_mteam())
        for func in (self._scrape_javbus, self._scrape_javdb, self._scrape_github):
            result.update(self._normalize_id(func()))

        uniq_id = len(result)
        prefix_counter = Counter(map(itemgetter(0), result))
        result = {k for k, v in prefix_counter.items() if v >= _THRESH}

        print(f"Uniq ID: {uniq_id}. Uniq prefix: {len(prefix_counter)}. Final: {len(result)}.")
        return result

    @staticmethod
    def _normalize_id(wordlist: Iterable[str]) -> Iterator[Tuple[str, str]]:
        matcher = re.compile(r"\s*([a-z]{3,7})[ _-]?0*([0-9]{2,6})\s*")
        for m in filter(None, map(matcher.fullmatch, map(str.lower, wordlist))):
            yield m.group(1, 2)

    def _scrape_mteam(self) -> Iterator[Tuple[str, str]]:

        print("Scanning mteam...")

        matcher = re.compile(
            r"(?:^|/)(?:[0-9]{3})?([a-z]{3,6})-0*([0-9]{2,4})(?:hhb[1-9]?)?\b.*\.(?:mp4|wmv|avi|mkv|iso)$",
            flags=re.MULTILINE,
        ).search

        for path in self.mteam_scrape:
            with open(path, "r", encoding="utf-8") as f:
                for m in filter(None, map(matcher, f)):
                    yield m.group(1, 2)

    @classmethod
    def _scrape_javbus(cls) -> Iterator[str]:

        print("Scanning javbus...")
        xpath = XPath('//div[@id="waterfall"]//a[@class="movie-box"]//span/date[1]/text()')
        step = 500

        with ThreadPoolExecutor(max_workers=None) as ex:
            for base in ("page", "uncensored/page", "genre/hd", "uncensored/genre/hd"):
                idx = 1
                print(f"  /{base}: ", end="", flush=True)

                while True:
                    print(f"{idx}:{idx+step}...", end="", flush=True)
                    args = ((f"https://www.javbus.com/{base}/{i}", xpath) for i in range(idx, idx + step))
                    try:
                        yield from chain.from_iterable(ex.map(cls._scrap_page, args))
                    except LastPageReached:
                        break
                    idx += step
                print()

    @classmethod
    def _scrape_javdb(cls) -> Iterator[str]:

        print(f"Scanning javdb...")
        xpath = XPath('//div[@id="videos"]//a/div[@class="uid"]/text()')
        args = ((f"https://javdb.com/{p}?page={i}", xpath) for p in ("uncensored", "") for i in range(1, 81))

        with ThreadPoolExecutor(max_workers=3) as ex:
            for future in as_completed(ex.submit(cls._scrap_page, i) for i in args):
                try:
                    for i in future.result():
                        yield i
                except LastPageReached:
                    pass

    @staticmethod
    def _scrap_page(args: Tuple[str, XPath]) -> List[str]:

        url, xpath = args

        for _ in range(3):
            try:
                res = session.get(url, timeout=(7, 27))
                res.raise_for_status()
            except requests.HTTPError as e:
                if e.response.status_code == 404:
                    raise LastPageReached
            except requests.RequestException:
                pass
            else:
                return xpath(fromstring(res.content))
            sleep(1)

        raise requests.RequestException(f"Connection error: {url}")

    @staticmethod
    def _scrape_github() -> List[str]:

        url = "https://raw.githubusercontent.com/imfht/fanhaodaquan/master/data/codes.json"
        print("Downloading github database...")

        for retry in range(3):
            try:
                return session.get(url, timeout=(7, 27)).json()
            except requests.RequestException:
                if retry == 2:
                    raise
                sleep(1)

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


class Analyzer:
    def __init__(self, report_dir: Path, regex_file: Path, scraper: MteamScraper) -> None:

        self.av_matcher = re.compile(regex_file.read_text(encoding="utf-8").strip()).search
        self.video_filter = re.compile(
            r"\.(?:3gp|asf|avi|bdmv|flv|iso|m(?:2?ts|4p|[24kop]v|p2|p4|pe?g|xf)|rm|rmvb|ts|vob|webm|wmv)$",
            flags=re.M,
        ).search

        report_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir = report_dir
        self.scraper = scraper

    def analyze_av(self, lo: int, hi: int):

        print("Matching test begins...")

        page = "adult.php"
        unmatch_raw = self.report_dir.joinpath("unmatch_raw.txt")
        unmatch_freq = self.report_dir.joinpath("unmatch_frequency.txt")
        total = unmatched = 0
        sep = "-" * 80 + "\n"

        prefix_searcher = re.compile(r"\b[0-9]{,3}([a-z]{2,8})[ _-]?[0-9]{2,6}(?:hhb[0-9]?)?\b").search
        word_finder = re.compile(r"(?!\d+\b)\w{3,}").findall

        flat_counter = defaultdict(list)
        prefix_counter = Counter()
        word_counter = Counter()
        tmp = set()

        with ProcessPoolExecutor(max_workers=None) as ex, unmatch_raw.open("w", encoding="utf-8") as f:

            paths = self.scraper.run(page, lo, hi, is_av=True)

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
        result = [(i, len(v), k, set(v)) for k, v in flat_counter.items() if (i := prefix_counter[k]) >= _THRESH]
        result.sort(reverse=True)
        words = [(v, k) for k, v in word_counter.items() if v >= _THRESH]
        words.sort(reverse=True)

        with unmatch_freq.open("w", encoding="utf-8") as f:
            f.write(stat + "\n\n")

            f.write("Potential ID Prefixes:\n\n")
            f.writelines(self._format_report(result))

            f.write("\n\nPotential Keywords:\n\n")
            f.write("{:>6}  {}\n{:->80}\n".format("uniq", "word", ""))
            f.writelines(f"{i:6d}  {j}\n" for i, j in words)

        print(stat)
        print(f"Result saved to: {unmatch_freq}")

    def analyze_non_av(self, lo: int, hi: int):

        print("Mismatch testing begins...")

        page = "torrents.php"
        mismatched_file = self.report_dir.joinpath("mismatch_frequency.txt")
        word_searcher = re.compile(r"[a-z]+").search
        total = mismatched = 0

        flat_counter = defaultdict(list)
        torrent_counter = Counter()
        tmp = set()

        with ProcessPoolExecutor(max_workers=None) as ex:

            paths = self.scraper.run(page, lo, hi, is_av=False)

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

        with mismatched_file.open("w", encoding="utf-8") as f:
            f.write(stat + "\n\n")
            f.writelines(self._format_report(result))

        print(stat)
        print(f"Result saved to: {mismatched_file}")

    def _match_av(self, path: Path) -> Optional[List[str]]:
        result = []
        matcher = self.av_matcher
        with open(path, "r", encoding="utf-8") as f:
            for v in filter(self.video_filter, f):
                if matcher(v):
                    return
                result.append(v)
        return result

    def _match_non_av(self, path: Path) -> Tuple[str]:
        with open(path, "r", encoding="utf-8") as f:
            return tuple(m[0] for m in map(self.av_matcher, filter(self.video_filter, f)) if m)

    @staticmethod
    def _get_stat(name: str, total: int, n: int):
        return f"Total: {total}. {name}: {n}. Percentage: {n / total * 100:.2f}%."

    @staticmethod
    def _format_report(result: Iterable):
        yield "{:>6}  {:>6}  {:15}  {}\n{:->80}\n".format("uniq", "occur", "word", "strings", "")
        for i, j, k, s in result:
            yield f"{i:6d}  {j:6d}  {k:15}  {_repr(s)[1:-1]}\n"


def parse_config(configfile: str):

    parser = ConfigParser()

    if parser.read(configfile):
        return parser["DEFAULT"]

    parser["DEFAULT"] = {
        "output_file": "",
        "cache_dir": "",
        "mteam_username": "",
        "mteam_password": "",
        "mteam_page": "adult.php?cat410=1&cat429=1&cat426=1&cat437=1&cat431=1&cat432=1",
        "mteam_limit": "500",
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
        help="fetch id prefixes from web",
    )
    parser.add_argument(
        "range",
        nargs="*",
        action="store",
        type=int,
        help="range of mteam pages for testing, 1 or 2 integers",
        default=(0, 100),
    )

    args = parser.parse_args()

    if args.mode != "build":
        if len(args.range) == 1 and args.range[0] > 0:
            args.range.insert(0, 0)
        elif len(args.range) != 2 or args.range[0] >= args.range[1]:
            parser.error("Ranges should be 1 or 2 integers (low to high)")

    return args


def load_session(session_file):

    try:
        with open(session_file, "rb") as f:
            s: requests.Session = pickle.load(f)

        if not isinstance(s, requests.Session):
            raise ValueError

    except (OSError, pickle.PickleError, ValueError):
        s = requests.Session()
        s.cookies.set_cookie(create_cookie(domain="www.javbus.com", name="existmag", value="all"))
        s.headers.update(
            {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:80.0) Gecko/20100101 Firefox/80.0"}
        )

    global session
    session = s


def save_session(session_file):
    with open(session_file, "wb") as f:
        pickle.dump(session, f)


def main():

    chdir(Path(__file__).parent)

    config = parse_config("rebuilder.ini")
    args = parse_arguments()

    raw_dir = Path("raw")
    report_dir = Path("report")
    session_file = raw_dir.joinpath("cookies")
    output_file = Path(config["output_file"] or "regex.txt")

    load_session(session_file)
    mteam_scraper = MteamScraper(
        cache_dir=Path(config["cache_dir"] or "cache"),
        account=(config["mteam_username"], config["mteam_password"]),
        fetch=args.fetch,
    )

    if args.mode == "build":

        regex = JavREBuilder(
            output_file=output_file,
            raw_dir=raw_dir,
            mteam_scraper=mteam_scraper,
            mteam_page=config["mteam_page"],
            mteam_limit=config.getint("mteam_limit") or 500,
            fetch=args.fetch,
        ).run()

        if regex is not None:
            print(f"\nResult ({len(regex)} chars):")
            print(regex)

    else:

        analyzer = Analyzer(
            report_dir=report_dir,
            regex_file=output_file,
            scraper=mteam_scraper,
        )

        if args.mode == "test_match":
            analyzer.analyze_av(*args.range)
        else:
            analyzer.analyze_non_av(*args.range)

    save_session(session_file)


if __name__ == "__main__":
    main()
