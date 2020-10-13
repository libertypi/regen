#!/usr/bin/env python3

from sys import argv
from regenerator import Regen


def main():

    if len(argv) > 2:

        if argv[1] == "-e":
            for word in Regen(argv[2:]).to_text():
                print(word)
            return

        elif argv[1] == "-c":
            regen = Regen(argv[2:])
            print(regen.to_regex())
            regen.verify_result()
            return

        elif argv[1] == "-f":
            with open(argv[2], "r") as f:
                wordlist = f.read().splitlines()

            regen = Regen(wordlist)
            regex = regen.to_regex()

            print(regex)
            print("Length:", len(regex))

            regen.verify_result()
            return

    print("-e <regex>: extract regex\n-c <string>: compute regex\n-f <file>: compute regex from file")


if __name__ == "__main__":
    main()
