#!/usr/bin/env python3

import argparse
import os
import pickle
import re
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from configparser import ConfigParser
from itertools import chain, filterfalse, islice, tee
from operator import itemgetter, methodcaller
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

from regen import Regen

_THRESH = 5


class LastPageReached(Exception):
    pass


class Scraper:
    @classmethod
    def get_id(cls) -> Iterator[Tuple[str, str]]:
        matcher = re.compile(r"\s*([a-z]{3,8})[_-]?0*([1-9][0-9]{,5})\s*").fullmatch
        return map(itemgetter(1, 2), filter(None, map(matcher, map(str.lower, cls._scrape()))))

    @staticmethod
    def _scrape() -> Iterator[str]:
        raise NotImplemented

    @staticmethod
    def get_western() -> Iterator[str]:
        raise NotImplemented


class JavBusScraper(Scraper):
    @staticmethod
    def _scrape(
        base: str = "https://www.javbus.com/",
        paths: Iterable[str] = ("page", "uncensored/page", "genre/hd", "uncensored/genre/hd"),
    ) -> Iterator[str]:

        print(f"Scanning {base}...")
        parser = XPath('.//div[@id="waterfall"]//a[@class="movie-box"]//span/date[1]/text()', smart_strings=False)
        parser = _get_downloader(parser, raise_404=True)
        step = 500

        with ThreadPoolExecutor(max_workers=None) as ex:
            for path in paths:
                lo = 1
                print(f"  /{path}: ", end="", flush=True)

                while True:
                    hi = lo + step
                    print(f"{lo}:{hi}...", end="", flush=True)
                    urls = (f"{base}/{path}/{i}" for i in range(lo, hi))

                    try:
                        yield from chain.from_iterable(ex.map(parser, urls))
                    except LastPageReached:
                        break

                    lo = hi
                print()

    @classmethod
    def get_western(cls):
        matcher = re.compile(r"\s*(\w+)\.[0-2][0-9]\.(?:1[0-2]|0[1-9])\.(?:3[01]|[12][0-9]|0[1-9])\s*").fullmatch

        result = cls._scrape(
            base="https://www.javbus.org/",
            paths=("page", *(f"studio/{i}" for i in range(1, 5))),
        )
        result = Counter(m[1].lower() for m in map(matcher, result) if m)
        for k, v in result.items():
            if v > _THRESH:
                yield k


class JavDBScraper(Scraper):
    @staticmethod
    def _scrape(
        base=("uncensored", ""),
        limit: int = 81,
        xpath: str = './/div[@id="videos"]//a[@class="box"]/div[@class="uid"]/text()',
    ) -> Iterator[str]:

        print(f"Scanning javdb...")
        parser = _get_downloader(XPath(xpath, smart_strings=False))

        with ThreadPoolExecutor(max_workers=3) as ex:
            fts = as_completed(
                ex.submit(parser, f"https://javdb.com/{b}?page={i}") for b in base for i in range(1, limit)
            )
            yield from chain.from_iterable(map(methodcaller("result"), fts))

    @classmethod
    def get_western(cls):

        result = cls._scrape(
            base=("series/western",),
            limit=5,
            xpath='.//div[@id="series"]//div[@class="box"]/a[@title and strong and span]',
        )
        sub_nondigit = re.compile(r"\D+").sub
        matcher = re.compile(r"[\s\w]+").fullmatch
        sub_space = re.compile(r"\s+").sub

        for a in result:
            try:
                freq = int(sub_nondigit("", a.findtext("span")))
            except ValueError:
                continue
            if freq > _THRESH:
                title = a.findtext("strong")
                if matcher(title):
                    yield sub_space("", title).lower()


class GithubScraper(Scraper):
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
        self._limit = limit
        self._logined = False

    def get_id(self) -> Iterator[Tuple[str, str]]:

        id_finder = re.compile(
            r"""
            (?:^|/)(?:[0-9]{3})?
            ([a-z]{3,6})
            (-)?
            0*([1-9][0-9]{,3})
            (?(2)(?:hhb[0-9]*)?|hhb[0-9]*)
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

    def get_path(self, is_av: bool, fetch: bool = True, cjk_title_only: bool = False) -> Iterator[Path]:

        subdir = self._cache_dir.joinpath("av" if is_av else "non_av")
        subdir.mkdir(parents=True, exist_ok=True)

        if not fetch:
            matcher = re.compile(r"[0-9]+\.txt").fullmatch
            for path in subdir.iterdir():
                if matcher(path.name):
                    yield path
            return

        pool = []
        idx = 0
        joinpath = subdir.joinpath
        matcher = re.compile(r"\bid=([0-9]+)").search
        url = urljoin(self.DOMAIN, self._pages[is_av])
        step = min((os.cpu_count() + 4) * 3, 32 * 3, self._limit)

        if cjk_title_only:
            parser = self._get_cjk_parser()
        else:
            parser = XPath(
                './/form[@id="form_torrent"]//table[@class="torrentname"]'
                '/descendant::a[contains(@href, "download.php?")][1]/@href',
                smart_strings=False,
            )
        parser = _get_downloader(parser)

        print(f"Scanning mteam ({self._limit} pages)...", end="", flush=True)
        if not self._logined:
            self._login()

        with ThreadPoolExecutor(max_workers=None) as ex:

            for ft in as_completed(ex.submit(parser, url, params={"page": i}) for i in range(1, self._limit + 1)):

                idx += 1
                if not idx % step:
                    if idx <= self._limit - step:
                        print(f"{idx}..", end="", flush=True)
                    else:
                        print(f"{idx}..{self._limit}")

                for link in ft.result():
                    try:
                        path = joinpath(matcher(link)[1] + ".txt")
                    except TypeError:
                        continue
                    if path.exists():
                        yield path
                    else:
                        pool.append(ex.submit(self._dl_torrent, urljoin(self.DOMAIN, link), path))

            if pool:
                yield from filter(None, map(methodcaller("result"), as_completed(pool)))

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
        title_xp = XPath(
            'descendant::a[*//text() and contains(@href, "details.php?")][1]//text()',
            smart_strings=False,
        )
        link_xp = XPath(
            'descendant::a[contains(@href, "download.php?")][1]/@href',
            smart_strings=False,
        )

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
            for table in tree.iterfind('.//form[@id="form_torrent"]//table[@class="torrentname"]'):
                try:
                    if any(1 << ord(c) & cjk for c in title_xp(table)[0]):
                        result.extend(link_xp(table))
                except IndexError:
                    pass
            return result

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

        raw_dir = self._raw_dir
        wordlist_file = raw_dir.joinpath(f"{name}.txt")
        whitelist_file = raw_dir.joinpath(f"{name}_whitelist.txt")
        blacklist_file = raw_dir.joinpath(f"{name}_blacklist.txt")

        wordlist = _update_file(wordlist_file, getattr(self, f"_{name}_strategy"))

        whitelist = _update_file(whitelist_file, self._filter_regex)
        if whitelist:
            regex = re.compile(r"\b(?:{})\b".format("|".join(whitelist)))
            regex = regex.search if name == "keyword" else regex.fullmatch
            wordlist = filterfalse(regex, wordlist)
        wordlist = set(wordlist)
        wordlist.update(whitelist)

        blacklist = _update_file(blacklist_file, self._filter_regex)
        wordlist.difference_update(blacklist)
        if name != "keyword":
            blacklist.append(self.keyword_regex)
        if blacklist:
            wordlist = filterfalse(re.compile("|".join(blacklist)).fullmatch, wordlist)

        wordlist = sorted(wordlist)
        print(f"{name} chosen: {len(wordlist)}")

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
        print(f"{name} regex test passed, {-diff} characters saved.")
        return computed

    def _keyword_strategy(self, old_list: Iterable[str]) -> Iterable[str]:
        if not self._fetch and old_list:
            return self._filter_regex(old_list)
        return set(chain(JavBusScraper.get_western(), JavDBScraper.get_western()))

    def _prefix_strategy(self, old_list: Iterable[str]) -> Iterable[str]:
        if not self._fetch and old_list:
            return self._filter_regex(old_list)

        result = set(self._mteam.get_id())
        for cls in JavBusScraper, JavDBScraper, GithubScraper:
            result.update(cls.get_id())

        prefix_counter = Counter(map(itemgetter(0), result))
        print(f"Uniq ID: {len(result)}. Uniq prefix: {len(prefix_counter)}.")

        return (k for k, v in prefix_counter.items() if v >= _THRESH)

    @staticmethod
    def _filter_regex(wordlist: Iterable[str]) -> Set[str]:
        return set(map(str.lower, filter(None, map(str.strip, wordlist))))


def _update_file(file: Path, stragety: Callable[[Iterable[str]], Iterable[str]]) -> List[str]:

    try:
        with open(file, "r+", encoding="utf-8") as f:
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
        with open(file, mode="w", encoding="utf-8") as f:
            f.writelines(i + "\n" for i in result)
        print(f"{file} created.")

    return result


class Analyzer:

    _video_re = r"\.(?:m(?:p4|[24kop]v|2?ts|4p|p2|pe?g|xf)|wmv|avi|iso|3gp|asf|bdmv|flv|rm|rmvb|ts|vob|webm)$"

    def __init__(self, regex_file: str, report_dir: str, mteam: MTeamScraper, fetch: bool) -> None:

        self._report_dir = Path(report_dir)
        self._report_dir.mkdir(parents=True, exist_ok=True)

        self._mteam = mteam
        self._fetch = fetch

        regex = Path(regex_file).read_text(encoding="utf-8").strip()
        assert len(regex.splitlines()) == 1, "regex file should contain only one line"

        p = regex.index("(", 1)
        self._av_matcher = re.compile(f"{regex[:p]}(?P<m>{regex[p+1:]}", flags=re.M).search
        self._video_filter = re.compile(self._video_re, flags=re.M).search

    def analyze_av(self):

        print("Matching test begins...")

        unmatch_raw = self._report_dir.joinpath("unmatch_raw.txt")
        unmatch_freq = self._report_dir.joinpath("unmatch_frequency.txt")
        total = unmatched = 0
        sep = "-" * 80 + "\n"

        prefix_finder = re.compile(r"\b([0-9]{,3}([a-z]{2,8})-?[0-9]{2,8}(?:hhb[0-9]*)?)\b.*$", flags=re.M).findall
        word_finder = re.compile(r"(?![\d_]+\b)\w{3,}").findall

        flat_counter = defaultdict(list)
        prefix_counter = Counter()
        word_counter = Counter()
        tmp = set()

        with ProcessPoolExecutor(max_workers=None) as ex, open(unmatch_raw, "w", encoding="utf-8") as f:

            for content in ex.map(self._match_av, self._mteam.get_path(True, self._fetch), chunksize=100):

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

        stat = self._get_stat("Match", total, total - unmatched)
        freq_words = get_freq_words()

        result = [
            (i, len(v), k, set(v))
            for k, v in flat_counter.items()
            if (i := prefix_counter[k]) >= _THRESH and k not in freq_words
        ]
        result.sort(reverse=True)

        words = [(v, k) for k, v in word_counter.items() if v >= _THRESH and k not in freq_words]
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

            for video in ex.map(self._match_non_av, self._mteam.get_path(False, self._fetch), chunksize=100):

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

        stat = self._get_stat("Mismatch", total, mismatched)
        result = [(torrent_counter[k], len(v), k, set(v)) for k, v in flat_counter.items()]
        result.sort(reverse=True)

        with open(mismatched_file, "w", encoding="utf-8") as f:
            f.write(stat + "\n\n")
            f.writelines(self._format_report(result))

        print(stat)
        print(f"Result saved to: {mismatched_file}")

    def _match_av(self, path: Path) -> Optional[str]:
        with open(path, "r", encoding="utf-8") as f:
            a, b = tee(filter(self._video_filter, f))
            if not any(map(self._av_matcher, a)):
                return "".join(b)

    def _match_non_av(self, path: Path) -> Tuple[str]:
        with open(path, "r", encoding="utf-8") as f:
            return tuple(m["m"] for m in map(self._av_matcher, filter(self._video_filter, f)) if m)

    @staticmethod
    def _get_stat(name: str, total: int, n: int):
        return f"Total: {total}. {name}: {n}. Percentage: {n / total:.2%}."

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


def _get_downloader(func: Callable[[HtmlElement], List[str]], raise_404: bool = False):
    def downloader(url: str, **kwargs):
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

    return downloader


def get_freq_words(lb: int = 3, ub: int = 6, n: int = 1000):

    freq_words = _request(
        "https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english-usa.txt"
    ).text
    freq_words = islice(re.finditer(f"^[A-Za-z]{{{lb},{ub}}}$", freq_words, flags=re.M), n)
    freq_words = frozenset(map(str.lower, map(itemgetter(0), freq_words)))
    assert len(freq_words) == n, "fetching frequent words failed"

    return freq_words


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

    os.chdir(Path(__file__).parent)

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
