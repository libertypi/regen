#!/usr/bin/env python3

import os.path
import unittest

from regenerator import Regen, Parser


class Testregen(unittest.TestCase):
    def test_extractor(self):
        values = (
            (r"A[BC]D", ["ABD", "ACD"]),
            (r"A(B|CD)E", ["ABE", "ACDE"]),
            (r"A(B[CD]E|FG)H", ["ABCEH", "ABDEH", "AFGH"]),
            (r"A(B|(C[DE]|F)G)HI?", ["ABH", "ABHI", "ACDGH", "ACDGHI", "ACEGH", "ACEGHI", "AFGH", "AFGHI"]),
            (r"A[BCD]?", ["A", "AB", "AC", "AD"]),
            (r"A[BC]*D+E{2,5}F?", ["A[BC]*D+E{2,5}", "A[BC]*D+E{2,5}F"]),
            (r"sky(237)?", ["sky", "sky237"]),
            (r"[ _-]?", ["", " ", "-", "_"]),
            (r"A[B3-5]C", ["ABC", "A[3-5]C"]),
            (r"AB|C()D", ["AB", "CD"]),
            (r"(A(B|C(D|E)*)*|F)", ["A(B|C[DE]*)*", "F"]),
            (r"A(\B*\C|\D)?", ["A", "A\\B*\\C", "A\\D"]),
            (r"(ABC)*", ["(ABC)*"]),
            (r"((ABC)?)*", ["((ABC)?)*"]),
            (r"(A|B|C|D){3}", ["[ABCD]{3}"]),
            (r"(ABC|ABD|AB)+", ["(AB[CD]?)+"]),
            (r"AB[^CD]|AB[^]]|AB[]]*|ABE", ["ABE", "AB[]]*", "AB[^CD]", "AB[^]]"]),
        )
        for regex, answer in values:
            result = Regen([regex]).to_text()
            self.assertEqual(result, answer, msg=result)

    def test_extractor_raises(self):
        values = ("A[B", "A(BC|", "AB|((CD)", "A[[B]", "A[[BB]]", "A*?+B{}", "|{3}", "A|B{a3}", "A|B{3")
        for regex in values:
            with self.assertRaises(ValueError, msg=regex):
                Regen([regex]).to_text()

    def test_optimizer(self):
        values = (
            (("ACD", "ACE", "BCD", "BCE"), "[AB]C[DE]"),
            (
                (
                    "atkgirlfriend",
                    "tonightsgirlfriend",
                    "cruelgirlfriend",
                    "atkgirlfriends",
                    "tonightsgirlfriends",
                    "cruelgirlfriends",
                ),
                "(atk|cruel|tonights)girlfriends?",
            ),
            (("A|B*",), "(A|B*)"),
        )
        for wordlist, answer in values:
            self.assertEqual(Regen(wordlist).to_regex(), answer)

    def test_file(self):
        testFile = ("av_censored_id.txt", "av_uncensored_id.txt", "av_keyword.txt")
        testDir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../transmission-torrent-done/component/"))
        for file in testFile:
            file = os.path.join(testDir, file)
            try:
                with open(file, "r") as f:
                    wordlist = f.read().splitlines()
            except FileNotFoundError:
                print(file, "Not Found!")
            else:
                Regen(wordlist).verify_result()


if __name__ == "__main__":
    unittest.main()
