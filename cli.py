#!/usr/bin/env python3

from __init__ import Extractor, Optimizer


def main():
    from sys import argv

    if len(argv) > 2:
        if argv[1] == "-e":
            for string in argv[2:]:
                for word in Extractor(string).get_text():
                    print(word)
            return
        elif argv[1] == "-c":
            extractors = map(Extractor, argv[2:])
            regex = Optimizer(*extractors).result
            print(regex)
            return
        elif argv[1] == "-f":
            with open(argv[2], "r") as f:
                wordlist = f.read().splitlines()
            extractors = map(Extractor, wordlist)
            regex = Optimizer(*extractors).result
            print(regex)
            print("Length:", len(regex))
            return

    print("-e <regex>: extract regex\n-c <string>: compute regex\n-f <file>: compute regex from file")


if __name__ == "__main__":
    main()
