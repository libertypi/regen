import regenerator
import unittest
import os.path
import re


class Testregenerator(unittest.TestCase):
    def test_tokenizer(self):
        string = "ABCD*E?*+F{1,5}G"
        t = regenerator.Tokenizer(string)
        self.assertEqual(t.eat(), "A")
        self.assertEqual(t.peek(), "B")
        self.assertEqual(t.eat(), "B")
        self.assertEqual(t.peek(), "C")
        self.assertEqual(t.confirm(), "C")
        self.assertEqual(t.eat(), "D")
        self.assertEqual(t.eat_suffix(), "*")
        self.assertEqual(t.eat(), "E")
        self.assertEqual(t.eat_suffix(), "?*+")
        self.assertEqual(t.eat_suffix(), None)
        self.assertEqual(t.eat(), "F")
        self.assertEqual(t.eat_suffix(), "{1,5}")
        self.assertEqual(t.eat(), "G")

        self.assertRaises(RuntimeError, regenerator.Tokenizer(string).confirm)

    def test_extractor(self):
        values = (
            (r"A[BC]D", ("ABD", "ACD")),
            (r"A(B|CD)E", ("ABE", "ACDE")),
            (r"A(B[CD]E|FG)H", ("ABCEH", "ABDEH", "AFGH")),
            (r"A(B|(C[DE]|F)G)HI?", ("ABH", "ACDGH", "ACEGH", "AFGH", "ABHI", "ACDGHI", "ACEGHI", "AFGHI")),
            (r"A[BCD]?", ("A", "AB", "AC", "AD")),
            (r"A[BC]*D+E{2,5}F?", ("A[BC]*D+E{2,5}", "A[BC]*D+E{2,5}F")),
            (r"sky(237)?", ("sky", "sky237")),
            (r"[ _-]?", ("", " ", "_", "-")),
            (r"A[B3-5]C", ("ABC", "A[3-5]C")),
            (r"AB|CD", ("AB", "CD")),
            (r"(A(B|C(D|E)*)*|F)", ("A(B|C[DE]*)*", "F")),
            (r"A(\B*\C|\D)?", ("A", "A\\B*\\C", "A\\D")),
        )
        for regex, answer in values:
            result = tuple(regenerator.Extractor(regex).get_text())
            self.assertEqual(result, answer, msg=result)

    def test_extractor_raises(self):
        values = ("A[B", "A(BC|", "AB|((CD)", "A[[B]", "A[[BB]]", "A*?+B{}", "|{3}", "A|B{a3}", "A|B{3")
        for regex in values:
            with self.assertRaises(ValueError, msg=regex):
                tuple(regenerator.Extractor(regex).get_text())

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
        )
        for words, answer in values:
            e = (regenerator.Extractor(i) for i in words)
            self.assertEqual(regenerator.Optimizer(*e).result, answer)

    def test_file(self):
        testFile = ("av_regex/av_censored_id.txt", "av_regex/av_uncensored_id.txt")
        for file in testFile:
            file = os.path.join(os.path.dirname(__file__), file)
            if not os.path.exists(file):
                print(file, "Not Found!")
                continue

            with open(file) as f:
                words = f.read().splitlines()
            words.sort()

            e = (regenerator.Extractor(i) for i in words)
            r = regenerator.Optimizer(*e).result
            e = sorted(regenerator.Extractor(r).get_text())
            self.assertEqual(words, e)


def test_regex(regex: str, wordlist: list):
    extracted = regenerator.Extractor(regex).get_text()
    assert sorted(wordlist) == sorted(extracted), "Extracted regex is different from original words."

    regex = re.compile(regex)
    for i in wordlist:
        if not regex.fullmatch(i):
            assert re.search(r"[*+{}]", i), "Regex matching test failed."


if __name__ == "__main__":
    unittest.main()