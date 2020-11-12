#!/usr/bin/env python3

import argparse
import pickle
import re
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from configparser import ConfigParser
from itertools import chain, filterfalse
from operator import itemgetter
from os import chdir
from pathlib import Path
from typing import Callable, Iterable, Iterator, List, Set, TextIO, Tuple
from urllib.parse import urljoin

import requests
from lxml import etree, html
from torrentool.api import Torrent
from torrentool.exceptions import TorrentoolException

from regenerator import Regen


class LastPageReached(Exception):
    pass


class MteamScraper:

    DOMAIN = "https://pt.m-team.cc/"

    def __init__(self, cache_dir: Path, fetch: bool = False) -> None:

        self._cache_dir = cache_dir
        self._fetch = fetch

    def run(self, page: str, lo: int, hi: int, is_av: bool, cjk_only: bool = False) -> Iterator[TextIO]:

        subdir = self._cache_dir.joinpath("av" if is_av else "non_av")
        if not subdir.exists():
            subdir.mkdir(parents=True)

        if not self._fetch:
            f = None
            matcher = re.compile("[0-9]+\.txt").fullmatch

            for path in subdir.iterdir():
                if matcher(path.name):
                    with path.open("r", encoding="utf-8") as f:
                        yield f
            if f is not None:
                return

        page = self.DOMAIN + page
        self._id_searcher = re.compile(r"\bid=(?P<id>[0-9]+)").search
        self._tr_show = re.compile(r"^\s+(.+?) \([^)]+\)$", flags=re.MULTILINE).findall

        if cjk_only:
            parser = self._make_cjk_parser()
        else:
            parser = etree.XPath(
                '//*[@id="form_torrent"]/table[@class="torrents"]'
                '//*[@class="torrenttr"]/table[@class="torrentname"]'
                '//a[contains(@href, "download.php")]/@href'
            )

        with ThreadPoolExecutor(max_workers=None) as l_ex, ThreadPoolExecutor(max_workers=None) as t_ex:
            for l_ft in as_completed(l_ex.submit(self._get_link, page, i, parser) for i in range(lo, hi + 1)):
                for t_ft in as_completed(t_ex.submit(self._fetch_torrent, i, subdir) for i in l_ft.result()):
                    try:
                        with t_ft.result().open("r", encoding="utf-8") as f:
                            yield f
                    except AttributeError:
                        pass

    @staticmethod
    def _make_cjk_parser():

        table_xp = etree.XPath(
            '//*[@id="form_torrent"]/table[@class="torrents"]//*[@class="torrenttr"]/table[@class="torrentname"]'
        )
        title_xp = etree.XPath('(.//a[contains(@href, "details.php")]/@title)[1]')
        link_xp = etree.XPath('(.//a[contains(@href, "download.php")]/@href)[1]')
        cjk = (
            (4352, 4607),
            (11904, 42191),
            (43072, 43135),
            (44032, 55215),
            (63744, 64255),
            (65072, 65103),
            (65381, 65500),
            (131072, 196607),
        )

        def _parser(tree) -> List[str]:
            result = []
            for table in table_xp(tree):
                try:
                    if any(i <= c <= j for c in map(ord, title_xp(table)[0]) for i, j in cjk):
                        result.append(link_xp(table)[0])
                except IndexError:
                    pass
            return result

        return _parser

    @staticmethod
    def _get_link(page: str, n: int, parser: Callable) -> List[str]:
        print("Fetching page:", n)
        for retry in range(3):
            try:
                r = session.get(page, timeout=(7, 27), params={"page": n})
                r.raise_for_status()
            except requests.RequestException:
                if retry == 2:
                    raise
            else:
                return parser(html.fromstring(r.content))

    def _fetch_torrent(self, link: str, subdir: Path) -> Path:

        file = subdir.joinpath(self._id_searcher(link)["id"] + ".txt")

        if not file.exists():
            print("Fetching torrent:", link)

            for _ in range(3):
                try:
                    content = session.get(urljoin(self.DOMAIN, link), timeout=(7, 27)).content
                except (requests.RequestException, AttributeError):
                    pass
                else:
                    break
            else:
                print(f"Downloading torrent failed: {link}")
                return

            try:
                filelist = Torrent.from_string(content).files
                with open(file, "w", encoding="utf-8") as f:
                    f.writelines(i[0].lower() + "\n" for i in filelist)

            except (TorrentoolException, OSError, TypeError):

                torrent_file = file.with_suffix(".torrent")

                try:
                    torrent_file.write_bytes(content)
                    filelist = subprocess.check_output(("transmission-show", torrent_file), encoding="utf-8")
                    filelist = self._tr_show(filelist, filelist.index("\n\nFILES\n\n"))
                    if not filelist:
                        raise ValueError

                    with open(file, "w", encoding="utf-8") as f:
                        f.writelines(i.lower() + "\n" for i in filelist)

                except (subprocess.CalledProcessError, ValueError, OSError):
                    print(f'Parsing torrent error: "{link}"')
                    return

                finally:
                    torrent_file.unlink(missing_ok=True)

        return file


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

        if not raw_dir.exists():
            raw_dir.mkdir(parents=True)

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
            f = file.open(mode="r+", encoding="utf-8")
            old_list = f.read().splitlines()
        except FileNotFoundError:
            f = file.open(mode="w", encoding="utf-8")
            old_list = []
        finally:
            result = sorted(stragety(old_list))
            if old_list != result:
                f.seek(0)
                f.writelines(i + "\n" for i in result)
                f.truncate()
                print(f"{file} updated.")
            f.close()
        return result

    def _prefix_strategy(self, old_list: Iterable[str]) -> Iterator[str]:

        if self._fetch or not old_list:
            result = self._web_scrape()
        else:
            result = self._filter_strategy(old_list)

        result.update(self._update_file(self.whitelist_file, self._extract_strategy))
        result.difference_update(self._update_file(self.blacklist_file, self._extract_strategy))

        return filterfalse(self._kw_filter, result)

    def _extract_strategy(self, old_list: Iterable[str]) -> Iterator[str]:
        return Regen(self._filter_strategy(old_list)).to_text()

    @staticmethod
    def _filter_strategy(wordlist: Iterable[str]) -> Set[str]:
        return set(map(str.lower, filter(None, map(str.strip, wordlist))))

    def _web_scrape(self) -> Set[str]:

        uniq_id = set(self._scrape_mteam())
        uniq_id.update(self._normalize_id(chain(self._scrape_javbus(), self._scrape_javdb(), self._scrape_github())))

        prefix_counter = Counter(map(itemgetter(0), uniq_id))
        final = {k for k, v in prefix_counter.items() if v >= 5}

        print(f"Uniq ID: {len(uniq_id)}. Uniq prefix: {len(prefix_counter)}. Final: {len(final)}.")
        return final

    @staticmethod
    def _normalize_id(wordlist: Iterable[str]) -> Iterator[Tuple[str, str]]:

        matcher = re.compile(r"\s*([a-z]{3,7})[ _-]?0*([0-9]{2,6})\s*").fullmatch
        for m in filter(None, map(matcher, map(str.lower, wordlist))):
            yield m.group(1, 2)

    def _scrape_mteam(self) -> Iterator[Tuple[str, str]]:

        print(f"Scanning MTeam...")

        matcher = re.compile(
            r"(?:^|/)(?:[0-9]{3})?([a-z]{3,6})-0*([0-9]{2,4})(?:hhb[0-9]?)?\b.*\.(?:mp4|wmv|avi|iso)$",
            flags=re.MULTILINE,
        ).search

        for m in filter(None, map(matcher, chain.from_iterable(self.mteam_scrape))):
            yield m.group(1, 2)

    def _scrape_javbus(self) -> Iterator[str]:

        print("Scanning javbus...")
        xpath = etree.XPath('//div[@id="waterfall"]//a[@class="movie-box"]//span/date[1]/text()')
        step = 500

        for base in ("page", "uncensored/page", "genre/hd", "uncensored/genre/hd"):
            idx = 1
            print(f"  /{base}: ", end="", flush=True)
            with ThreadPoolExecutor(max_workers=None) as ex:
                while True:
                    print(f"{idx}:{idx+step}...", end="", flush=True)
                    args = ((f"https://www.javbus.com/{base}/{i}", xpath) for i in range(idx, idx + step))
                    try:
                        yield from chain.from_iterable(ex.map(self._scrap_jav, args))
                    except LastPageReached:
                        break
                    idx += step
            print()

    def _scrape_javdb(self) -> Iterator[str]:

        limit = 80
        print(f"Scanning javdb...")
        xpath = etree.XPath('//*[@id="videos"]//a/div[@class="uid"]/text()')

        for base in ("https://javdb.com/uncensored", "https://javdb.com/"):
            with ThreadPoolExecutor(max_workers=3) as ex:
                args = ((f"{base}?page={i}", xpath) for i in range(1, limit + 1))
                try:
                    yield from chain.from_iterable(ex.map(self._scrap_jav, args))
                except LastPageReached:
                    pass

    @staticmethod
    def _scrap_jav(args: Tuple) -> List[str]:

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
                return xpath(html.fromstring(res.content))

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

        self.av_matcher = re.compile(regex_file.read_text(encoding="utf-8").strip(), flags=re.M).search

        if not report_dir.exists():
            report_dir.mkdir(parents=True)

        self.report_dir = report_dir
        self.scraper = scraper

    def analyze_av(self, lo: int, hi: int):

        print("Matching test begins...")

        page = "adult.php"
        unmatched_file = self.report_dir.joinpath("unmatch_raw.txt")
        freq_file = self.report_dir.joinpath("unmatch_frequency.txt")
        total = unmatched = 0
        sep = "-" * 80 + "\n"

        av_matcher = self.av_matcher
        prefix_searcher = re.compile(r"\b[0-9]*([a-z]{2,8})[ _-]?[0-9]{2,6}(?:hhb[0-9]?)?\b").search
        word_finder = re.compile(r"\w{3,}").findall

        flat_counter = defaultdict(set)
        prefix_counter = Counter()
        word_counter = Counter()
        videos = []
        tmp = set()

        with unmatched_file.open("w", encoding="utf-8") as f:

            for t in self.scraper.run(page, lo, hi, is_av=True):

                total += 1
                if any(map(av_matcher, t)):
                    continue

                t.seek(0)
                videos.extend(filter(is_video, t))
                if not videos:
                    continue

                unmatched += 1
                f.write(sep)
                f.writelines(videos)
                f.write("\n")

                for m in filter(None, map(prefix_searcher, videos)):
                    prefix = m.group(1)
                    flat_counter[prefix].add(m.group())
                    tmp.add(prefix)

                prefix_counter.update(tmp)
                tmp.clear()

                tmp.update(chain.from_iterable(map(word_finder, videos)))
                word_counter.update(tmp)
                tmp.clear()
                videos.clear()

            f.write(f"{sep}Total: {total}. Unmatched: {unmatched}.\n")

        prefixes = [(i, len(v), k, v) for k, v in flat_counter.items() if (i := prefix_counter[k]) >= 3]
        words = [(v, k) for k, v in word_counter.items() if v >= 3]
        prefixes.sort(reverse=True)
        words.sort(reverse=True)

        with freq_file.open("w", encoding="utf-8") as f:
            f.write("Potential ID Prefixes:\n\n")
            f.write("{:>6}  {:>6}  {:15}  {}\n{:->80}\n".format("uniq", "occur", "word", "strings", ""))
            f.writelines(f"{i:6d}  {j:6d}  {k:15}  {s}\n" for i, j, k, s in prefixes)

            f.write("\n\nPotential Keywords:\n\n")
            f.write("{:>6}  {}\n{:->80}\n".format("uniq", "word", ""))
            f.writelines(f"{i:6d}  {j}\n" for i, j in words)

        print(f"Done. Results saved to:\n{unmatched_file}\n{freq_file}")

    def analyze_non_av(self, lo: int, hi: int):

        print("Mismatch testing begins...")

        page = "torrents.php"
        mismatched_file = self.report_dir.joinpath("mismatch_frequency.txt")
        word_searcher = re.compile(r"[a-z]+").search
        av_matcher = self.av_matcher

        flat_counter = defaultdict(list)
        torrent_counter = Counter()
        tmp = set()

        for f in self.scraper.run(page, lo, hi, is_av=False):

            for m in filter(None, map(av_matcher, f)):
                m = m.group()
                try:
                    word = word_searcher(m).group()
                except AttributeError:
                    pass
                else:
                    flat_counter[word].append(m)
                    tmp.add(word)

            if tmp:
                torrent_counter.update(tmp)
                tmp.clear()

        result = [(torrent_counter[k], len(v), k, set(v)) for k, v in flat_counter.items()]
        result.sort(reverse=True)

        with mismatched_file.open("w", encoding="utf-8") as f:
            f.write("{:>6}  {:>6}  {:15}  {}\n{:->80}\n".format("uniq", "occur", "word", "strings", ""))
            f.writelines(f"{i:6d}  {j:6d}  {k:15}  {s}\n" for i, j, k, s in result)

        print(f"Done. Result saved to: {mismatched_file}")


def is_video(string: str):
    return string.rstrip().endswith((".mp4", ".wmv", ".avi", ".iso", ".m2ts"))


def parse_config(configfile: str):

    parser = ConfigParser()

    if parser.read(configfile):
        return parser["DEFAULT"]

    parser["DEFAULT"] = {
        "output_file": "",
        "cache_dir": "",
        "session_file": "",
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
        default=(0, 20),
    )

    args = parser.parse_args()

    if args.mode != "build":
        if len(args.range) == 1 and args.range[0] > 0:
            args.range.insert(0, 0)
        elif len(args.range) != 2 or args.range[0] >= args.range[1]:
            parser.error("Ranges should be 1 or 2 integers (low to high)")

    return args


def read_session(session_file: Path):

    with open(session_file, "rb") as f:
        s: requests.Session = pickle.load(f)[4]
    s.cookies.set_cookie(requests.cookies.create_cookie(domain="www.javbus.com", name="existmag", value="all"))

    global session
    session = s


def main():

    chdir(Path(__file__).parent)

    config = parse_config("rebuilder.ini")
    args = parse_arguments()

    read_session(Path(config["session_file"] or None))
    output_file = Path(config["output_file"] or "regex.txt")

    mteam_scraper = MteamScraper(
        cache_dir=Path(config["cache_dir"] or "cache"),
        fetch=args.fetch,
    )

    if args.mode == "build":

        regex = JavREBuilder(
            output_file=output_file,
            raw_dir=Path("raw"),
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
            report_dir=Path("report"),
            regex_file=output_file,
            scraper=mteam_scraper,
        )

        if args.mode == "test_match":
            analyzer.analyze_av(*args.range)
        else:
            analyzer.analyze_non_av(*args.range)


if __name__ == "__main__":
    main()
