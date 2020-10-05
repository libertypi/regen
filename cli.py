#!/usr/bin/env python3

from __init__ import Extractor, Optimizer


def main():
    from sys import argv, exit

    if len(argv) > 2:
        if argv[1] == "-e":
            for string in argv[2:]:
                for word in Extractor(string).get_text():
                    print(word)
            exit()
        elif argv[1] == "-c":
            extractors = (Extractor(i) for i in argv[2:])
            regex = Optimizer(*extractors).result
            print(regex)
            exit()
        elif argv[1] == "-f":
            with open(argv[2], "r") as f:
                wordlist = f.read().splitlines()
            extractors = (Extractor(i) for i in wordlist)
            regex = Optimizer(*extractors).result
            print(regex)
            print("Length:", len(regex))
            exit()

    print("-e <regex>: extract regex\n-c <string>: compute regex\n-f <file>: compute regex from file")


if __name__ == "__main__":
    main()
