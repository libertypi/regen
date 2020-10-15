#!/usr/bin/env python3

"""
__main__.py
Module for command-line execution.
"""

import argparse

from . import Regen


def parse_arguments():

    parser = argparse.ArgumentParser(
        prog="ReGenerator",
        description="Generate regular expressions from a set of words and regexes.",
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "-e",
        "--extract",
        dest="mode",
        action="store_const",
        const="extract",
        help="extract Regex to a list of corresponding words.",
    )
    mode_group.add_argument(
        "-c",
        "--compute",
        dest="mode",
        action="store_const",
        const="compute",
        help="compute an optimized regex matching the given text.",
    )
    parser.set_defaults(mode="compute")

    parser.add_argument(
        "-v",
        "--verify",
        dest="verify",
        action="store_true",
        help="verify the generated regex.",
    )
    parser.add_argument(
        "-o",
        "--omit",
        dest="omit",
        action="store_true",
        help="omit the outer parentheses, if any.",
    )

    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "-f",
        "--file",
        dest="file",
        type=argparse.FileType("r"),
        help="take text from FILE, one word per line.",
    )
    target_group.add_argument(
        dest="word",
        nargs="*",
        action="store",
        default=(),
        help="a list of words/regexes, one word per argument.",
    )

    return parser.parse_args()


def main():

    args = parse_arguments()

    if args.file is None:
        wordlist = args.word
    else:
        wordlist = args.file.read().splitlines()
        args.file.close()

    regen = Regen(wordlist)

    if args.mode == "extract":
        for word in regen.to_text():
            print(word)

    else:
        regex = regen.to_regex(omitOuterParen=args.omit)
        print(regex)

        if args.verify:
            print("\nLength:", len(regex))
            print("Verifying... ", end="")
            try:
                regen.verify_result()
            except AssertionError as e:
                print("failed:", e)
            else:
                print("passed.")


if __name__ == "__main__":
    main()
