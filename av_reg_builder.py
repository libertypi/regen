#!/usr/bin/env python3

import os.path
import test

import regenerator


def read_file(file: str, extractWriteBack: bool = False):
    path = os.path.join(os.path.dirname(__file__), file)

    with open(path, mode="r+", encoding="utf-8") as f:

        o_list = f.read().splitlines()
        s_list = {i.lower() for i in o_list if i}

        extracters = tuple(regenerator.Extractor(i) for i in s_list)
        if extractWriteBack:
            s_list = [j for i in extracters for j in i.get_text()]
        else:
            s_list = list(s_list)
        s_list.sort()

        if o_list != s_list:
            f.seek(0)
            f.write("\n".join(s_list))
            f.write("\n")
            f.truncate()
            print(f"{file} updated.")

    return extracters, s_list


def write_file(file: str, content: str, checkDiff: bool = True):
    path = os.path.join(os.path.dirname(__file__), file)

    try:
        with open(path, mode="r", encoding="utf-8") as f:
            old = f.read()
        if checkDiff and old == content:
            print(f"{file} skiped.")
            return
    except FileNotFoundError:
        pass

    with open(path, mode="w", encoding="utf-8") as f:
        f.write(content)
        if not content.endswith("\n"):
            f.write("\n")

        print(f"{file} updated.")


def optimize_regex(extractors: list, wordlist: list, unittest=True):
    computed = regenerator.Optimizer(*extractors).result

    if unittest:
        test.test_regex(regex=computed, wordlist=wordlist)

    return computed


def main():
    kwExtractors, kwList = read_file("av_regex/av_keyword.txt", extractWriteBack=False)

    cidExtractors, cidList = read_file("av_regex/av_censored_id.txt", extractWriteBack=True)
    ucidExtractors, ucidList = read_file("av_regex/av_uncensored_id.txt", extractWriteBack=True)

    remove = set(ucidList)
    intersect = remove.intersection(cidList)
    if intersect:
        remove.difference_update(intersect)
        ucidList = sorted(remove)
        ucidExtractors = tuple(regenerator.Extractor(i) for i in remove)
        write_file("av_regex/av_uncensored_id.txt", "\n".join(ucidList), checkDiff=False)

    av_keyword = "|".join(kwList)
    av_censored_id = optimize_regex(cidExtractors, cidList)
    av_uncensored_id = optimize_regex(ucidExtractors, ucidList)

    avReg = f"(^|[^a-z0-9])({av_keyword}|{av_uncensored_id}[ _-]*[0-9]{{2,6}}|[0-9]{{,4}}{av_censored_id}[ _-]*[0-9]{{2,6}})([^a-z0-9]|$)"
    write_file("av_regex/av_regex.txt", avReg, checkDiff=True)

    print("Regex:")
    print(avReg)
    print("Length:", len(avReg))


if __name__ == "__main__":
    main()
