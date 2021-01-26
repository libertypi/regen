#!/usr/bin/env python3

import argparse
import concurrent.futures as cf
import json
import os
import os.path as op
import re
import reprlib
import subprocess
import sys
from collections import Counter, defaultdict
from configparser import ConfigParser
from itertools import chain, filterfalse, islice, repeat, tee
from operator import itemgetter, methodcaller
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, Union
from urllib.parse import urljoin, urlsplit

import requests
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

    def get_id(self) -> Iterable[str]:
        raise NotImplementedError

    def get_keyword(self) -> Iterable[Tuple[str, int]]:
        raise NotImplementedError

    def _scrape(self, base: str, paths: Tuple[str], xpath: Union[XPath, str],
                step: int) -> Iterator[str]:

        print(f"Scanning {urlsplit(base).netloc}")
        if isinstance(xpath, str):
            xpath = XPath(xpath, smart_strings=False)

        with cf.ThreadPoolExecutor() as ex:
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

    def __init__(self) -> None:

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

    def get_id(self):

        return self._scrape(
            base="https://www.javbus.com",
            paths=("/page", "/uncensored/page", "/genre/hd",
                   "/uncensored/genre/hd"),
            xpath=self.xpath,
            step=500,
        )

    def get_keyword(self):

        result = self._scrape(
            base="https://www.javbus.org",
            paths=("/page", *(f"/studio/{i}" for i in range(1, 5))),
            xpath=self.xpath,
            step=500,
        )
        matcher = re.compile(r"\s*([A-Za-z0-9]{3,})(?:\.\d\d){3}\s*").fullmatch
        return Counter(m[1].lower() for m in map(matcher, result) if m).items()


class JavDBScraper(Scraper):

    DOMAIN = "https://javdb.com"

    def __init__(self) -> None:

        self.nav_xp = XPath(
            '//section/div[@class="container"]//ul[@class="pagination-list"]/li'
            '/a[contains(@class, "is-current") and normalize-space()=$page]')

    def get_tree(self, url: str, page: int):

        tree = html_fromstring(_request(f"{url}?page={page}").content)
        if self.nav_xp(tree, page=page):
            return tree
        raise LastPageReached(page)

    def get_id(self) -> Iterator[str]:

        return self._scrape(
            base=self.DOMAIN,
            paths=("/uncensored", "/"),
            xpath=('.//div[@id="videos"]//a[@class="box"]'
                   '/div[@class="uid"]/text()'),
            step=100,
        )

    def get_keyword(self) -> Iterable[Tuple[str, int]]:

        result = self._scrape(
            base=self.DOMAIN,
            paths=("/series/western",),
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


class GithubScraper:

    @staticmethod
    def get_id() -> List[str]:

        print("Downloading GitHub database...")
        return _request(
            "https://raw.githubusercontent.com/imfht/fanhaodaquan/master/data/codes.json"
        ).json()


class MGSLoader:

    def __init__(self, json_file: str) -> None:
        self.json_file = Path(json_file)

    def get_id(self) -> List[str]:

        print("Loading MGStage data...")
        try:
            with open(self.json_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError) as e:
            print(e, file=sys.stderr)
            return ()


class Builder:

    def __init__(self, *, output_file: str, keyword_max: int, prefix_max: int,
                 mgs_json: str, raw_dir: str) -> None:

        self._output_file = Path(output_file)
        self._raw_dir = Path(raw_dir)
        self._jsonfile = self._raw_dir.joinpath("data.json")
        self._keyword_max = keyword_max
        self._prefix_max = prefix_max
        self._mgs_json = mgs_json
        self._scrapers = None

    def from_web(self):

        if not self._scrapers:
            self._scrapers = [JavBusScraper(), JavDBScraper(), GithubScraper]
            if self._mgs_json:
                self._scrapers.append(MGSLoader(self._mgs_json))

        data = {
            "keyword": self._fetch_keyword(),
            "prefix": self._fetch_prefix()
        }
        with open(self._jsonfile, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

        return self._build(data)

    def from_cache(self):

        try:
            with open(self._jsonfile, "r", encoding="utf-8") as f:
                data = json.load(f)
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
            regex_filter=keyword,
        )

        print("-" * 50)
        if not (keyword and prefix):
            return

        regex = rf"(^|[^a-z0-9])({keyword}|[0-9]{{,3}}{prefix}[_-]?[0-9]{{2,8}})([^a-z0-9]|$)"
        self._update_file(self._output_file, regex)
        return regex

    def _build_regex(self,
                     name: str,
                     data: Dict[str, dict],
                     omitOuterParen: bool,
                     regex_filter: str = None) -> str:

        print(f" {name.upper()} ".center(50, "-"))
        data = data[name]
        print(f"Total: {sum(data.values())}, unique: {len(data)}")

        words = sorted(data, key=data.get, reverse=True)
        i = getattr(self, f"_{name}_max")
        if 0 < i < len(words):
            thresh = data[words[i]]
            for i in range(i + 1, len(words)):
                if data[words[i]] < thresh:
                    words = words[:i]
                    break

        print("Cut: {} (frequency: {})".format(
            len(words), data[words[-1]] if words else None))

        whitelist = self._update_file(
            self._raw_dir.joinpath(f"{name}_whitelist.txt"))
        blacklist = self._update_file(
            self._raw_dir.joinpath(f"{name}_blacklist.txt"))

        if regex_filter:
            regex = chain(whitelist, blacklist, (regex_filter,))
        else:
            regex = chain(whitelist, blacklist)
        regex = re.compile("|".join(regex)).fullmatch
        words = set(filterfalse(regex, words))

        words.update(whitelist)
        words.difference_update(blacklist)
        words = sorted(words)

        print(f"Final: {len(words)}")

        regen = Regen(words)
        regex = regen.to_regex(omitOuterParen=omitOuterParen)

        concat = "|".join(words)
        if not omitOuterParen and len(words) > 1:
            concat = f"({concat})"

        length = len(regex)
        diff = length - len(concat)
        if diff > 0:
            print(
                f"Computed regex is {diff} characters longer than concatenation, use the latter."
            )
            regex = concat
        else:
            regen.raise_for_verify()
            print(f"Regex length: {length} ({diff})")

        return regex

    def _fetch_keyword(self) -> Dict[str, int]:

        result = {}
        setdefault = result.setdefault
        freq = get_freq_words()

        for s in self._scrapers:
            if not hasattr(s, "get_keyword"):
                continue
            for k, v in s.get_keyword():
                if k not in freq and setdefault(k, v) < v:
                    result[k] = v

        return self._sort_dict_by_val(result)

    def _fetch_prefix(self) -> Dict[str, int]:

        result = defaultdict(set)
        reg = re.compile(r"\s*\d*([a-z]{3,8})[_-]?(\d{2,8})\s*").fullmatch

        for s in self._scrapers:
            for m in filter(None, map(reg, map(str.lower, s.get_id()))):
                result[m[1]].add(int(m[2]))

        return self._sort_dict_by_val((k, len(v)) for k, v in result.items())

    @staticmethod
    def _sort_dict_by_val(d: Union[Dict, Iterable[Tuple]], reverse=True):
        if isinstance(d, dict):
            d = d.items()
        return dict(sorted(d, key=itemgetter(1), reverse=reverse))

    @staticmethod
    def _update_file(file: Path, content: str = None) -> List[str]:

        new = [] if content is None else [content]
        try:
            with open(file, "r+", encoding="utf-8") as f:
                old = f.read().splitlines()
                if not new:
                    new.extend(
                        set(map(str.lower, filter(None, map(str.strip, old)))))
                    new.sort()
                if old != new:
                    f.seek(0)
                    f.writelines(i + "\n" for i in new)
                    f.truncate()
                    print(f"Update: {file}")
        except FileNotFoundError:
            file.parent.mkdir(parents=True, exist_ok=True)
            with open(file, mode="w", encoding="utf-8") as f:
                f.writelines(i + "\n" for i in new)
            print(f"Create: {file}")
        return new


class MTeamCollector:

    DOMAIN = "https://pt.m-team.cc/"

    def __init__(self, page_max: int, av_page: str, non_av_page: str,
                 username: str, password: str, cache_dir: str) -> None:

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

    def get_path(self, is_av: bool, fetch: bool = True) -> Iterator[str]:

        cache_dir = self._cache_dirs[is_av]
        os.makedirs(cache_dir, exist_ok=True)

        if fetch:
            return self._from_web(cache_dir, self._pages[is_av])
        return self._from_cache(cache_dir)

    @staticmethod
    def _from_cache(cache_dir: str) -> Iterator[str]:

        matcher = re.compile(r"[0-9]+\.txt").fullmatch
        with os.scandir(cache_dir) as it:
            for entry in it:
                if matcher(entry.name):
                    yield entry.path

    def _from_web(self, cache_dir: str, url: str) -> Iterator[str]:

        pool = []
        idx = 0
        matcher = re.compile(r"\bid=([0-9]+)").search
        xpath = XPath(
            '//form[@id="form_torrent"]//table[@class="torrentname"]'
            '/descendant::a[contains(@href, "download.php?")][1]/@href',
            smart_strings=False)
        step = min((os.cpu_count() + 4) * 3, 32 * 3, self._page_max)
        join = op.join
        exists = op.exists

        print(f"Scanning mteam ({self._page_max} pages)...", end="", flush=True)
        self._login()

        with cf.ThreadPoolExecutor(max_workers=None) as ex:

            for ft in cf.as_completed(
                    ex.submit(_request, url, params={"page": i})
                    for i in range(1, self._page_max + 1)):

                idx += 1
                if not idx % step:
                    if idx <= self._page_max - step:
                        print(f"{idx}..", end="", flush=True)
                    else:
                        print(f"{idx}..{self._page_max}")

                for link in xpath(html_fromstring(ft.result().content)):
                    try:
                        path = join(cache_dir, matcher(link)[1] + ".txt")
                    except TypeError:
                        continue
                    if exists(path):
                        yield path
                    else:
                        pool.append(
                            ex.submit(self._dl_torrent,
                                      urljoin(self.DOMAIN, link), path))

            if pool:
                yield from filter(
                    None, map(methodcaller("result"), cf.as_completed(pool)))

    def _login(self):

        if self._logined:
            return
        try:
            res = session.post(
                url=self.DOMAIN + "takelogin.php",
                data=self._account,
                headers={"referer": self.DOMAIN + "login.php"},
            )
            res.raise_for_status()
            res = session.head(self._pages[0], allow_redirects=True)
            if "/login.php" in res.url:
                raise requests.RequestException("invalid credentials")
        except requests.RequestException as e:
            sys.exit(f"login mteam failed: {e}")
        self._logined = True

    @staticmethod
    def _dl_torrent(link: str, path: str):

        print("Downloading:", link)
        try:
            content = _request(link).content
        except requests.RequestException:
            print(f"Downloading failed: {link}", file=sys.stderr)
            return

        try:
            filelist = Torrent.from_string(content).files
            with open(path, "w", encoding="utf-8") as f:
                f.writelines(i[0].lower() + "\n" for i in filelist)

        except Exception:

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


class Analyzer:

    def __init__(self, *, regex_file: str, mteam: MTeamCollector, fetch: bool,
                 report_dir: str) -> None:

        self._mteam = mteam
        self._fetch = fetch
        self._report_dir = Path(report_dir)
        self._report_dir.mkdir(parents=True, exist_ok=True)

        try:
            with open(regex_file, "r", encoding="utf-8") as f:
                regex = f.readline().strip()
                if f.read():
                    raise ValueError
                i = regex.index("(", 1)
        except FileNotFoundError:
            sys.exit(f"{regex_file} not found")
        except ValueError:
            sys.exit("invalid regex file")

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

        with cf.ProcessPoolExecutor(max_workers=None) as ex, open(
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
                  if (i := prefix_counter[k]) >= 5 and k not in freq_words]
        result.sort(reverse=True)

        words = [(v, k)
                 for k, v in word_counter.items()
                 if v >= 5 and k not in freq_words]
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

        print("Mismatching test begins...")

        mismatched_file = self._report_dir.joinpath("mismatch_frequency.txt")
        word_searcher = re.compile(r"[a-z]+").search
        total = mismatched = 0

        flat_counter = defaultdict(list)
        torrent_counter = Counter()
        tmp = set()

        with cf.ProcessPoolExecutor(max_workers=None) as ex:

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
            yield f"{i:6d}  {j:6d}  {k:15}  {reprlib.repr(s)[1:-1]}\n"


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
                          backoff_factor=0.1))
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


def parse_config(configfile: str):

    parser = ConfigParser()
    if parser.read(configfile):
        config = parser["DEFAULT"]
        config["output_file"] = op.expanduser(config["output_file"])
        config["cache_dir"] = op.expanduser(config["cache_dir"])
        config["mgs_json"] = op.expanduser(config["mgs_json"])
        return config

    parser["DEFAULT"] = {
        "output_file":
            "regex.txt",
        "cache_dir":
            "cache",
        "mgs_json":
            "",
        "mteam_username":
            "",
        "mteam_password":
            "",
        "mteam_av_page":
            "adult.php?cat410=1&cat429=1&cat424=1&cat430=1&cat426=1&cat437=1&cat431=1&cat432=1",
        "mteam_non_av_page":
            "torrents.php",
    }
    with open(configfile, "w", encoding="utf-8") as f:
        parser.write(f)
    sys.exit(f"Please edit {configfile} before running me again.")


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
        const="test_miss",
        help="mis-matching test with non-av torrents",
    )
    group.set_defaults(mode="build")

    parser.add_argument(
        "-f",
        "--fetch",
        dest="fetch",
        action="store_true",
        help="fetch info from web instead of using local cache "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--pmax",
        dest="prefix_max",
        action="store",
        type=int,
        default=3000,
        help="maximum prefixes to use when building regex "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--kmax",
        dest="keyword_max",
        action="store",
        type=int,
        default=200,
        help="maximum keywords to use when building regex "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--mtmax",
        dest="mteam_max",
        action="store",
        type=int,
        default=500,
        help="maximum mteam pages to scrape (default: %(default)s)",
    )

    args = parser.parse_args()
    if args.mteam_max <= 0 or args.prefix_max <= 0:
        parser.error(
            "mteam_max and prefix_max should be an integer greater than zero")
    return args


session = _init_session()


def main():

    os.chdir(op.dirname(__file__))

    config = parse_config("builder.ini")
    args = parse_arguments()

    if args.mode == "build":

        builder = Builder(
            output_file=config["output_file"],
            keyword_max=args.keyword_max,
            prefix_max=args.prefix_max,
            mgs_json=config["mgs_json"],
            raw_dir="raw",
        )
        regex = builder.from_web() if args.fetch else builder.from_cache()

        if regex:
            print(f"\nResult ({len(regex)} chars):")
            print(regex)
        else:
            print("Generating regex failed.", file=sys.stderr)

    else:
        mteam = MTeamCollector(
            page_max=args.mteam_max,
            av_page=config["mteam_av_page"],
            non_av_page=config["mteam_non_av_page"],
            username=config["mteam_username"],
            password=config["mteam_password"],
            cache_dir=config["cache_dir"],
        )
        analyzer = Analyzer(
            regex_file=config["output_file"],
            mteam=mteam,
            fetch=args.fetch,
            report_dir="report",
        )
        if args.mode == "test_match":
            analyzer.analyze_av()
        else:
            analyzer.analyze_non_av()


if __name__ == "__main__":
    main()
