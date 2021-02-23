#!/usr/bin/env python3

import unittest

from regen import Regen


class Testregen(unittest.TestCase):

    def test_extractor(self):
        values = {
            'A[BC]D': ['ABD', 'ACD'],
            'A(B|CD)E': ['ABE', 'ACDE'],
            'A(B[CD]E|FG)H': ['ABCEH', 'ABDEH', 'AFGH'],
            'A(B|(C[DE]|F)G)HI?': [
                'ABH', 'ABHI', 'ACDGH', 'ACDGHI', 'ACEGH', 'ACEGHI', 'AFGH',
                'AFGHI'
            ],
            'A[BCD]?': ['A', 'AB', 'AC', 'AD'],
            'A[BC]*D+E{2,5}F?': ['A[BC]*D+E{2,5}', 'A[BC]*D+E{2,5}F'],
            'sky(237)?': ['sky', 'sky237'],
            '[ _-]?': ['', ' ', '-', '_'],
            'A[B3-5]C': ['ABC', 'A[3-5]C'],
            'AB|C()D': ['AB', 'CD'],
            '(A(B|C(D|E)*)*|F)': ['A(B|C[DE]*)*', 'F'],
            'A(\\B*\\C|\\D)?': ['A', 'A\\B*\\C', 'A\\D'],
            '(ABC)*': ['(ABC)*'],
            '((ABC)?)*': ['((ABC)?)*'],
            '(A|B|C|D){3}': ['[ABCD]{3}'],
            '(ABC|ABD|AB)+': ['(AB[CD]?)+'],
            'AB[^CD]|AB[^]]|AB[]]*|ABE': ['ABE', 'AB[]]*', 'AB[^CD]', 'AB[^]]']
        }
        for regex, answer in values.items():
            result = Regen([regex]).to_words()
            self.assertEqual(result, answer, msg=result)

    def test_extractor_raises(self):
        values = ("A[B", "A(BC|", "AB|((CD)", "A[[B]", "A[[BB]]", "A*?+B{}",
                  "|{3}", "A|B{a3}", "A|B{3")
        for regex in values:
            with self.assertRaises(ValueError, msg=regex):
                Regen([regex]).to_words()

    def test_optimizer(self):
        values = (
            (("ACD", "ACE", "BCD", "BCE"), "[AB]C[DE]"),
            (("atkgirlfriend", "tonightsgirlfriend", "cruelgirlfriend",
              "atkgirlfriends", "tonightsgirlfriends", "cruelgirlfriends"),
             "(atk|cruel|tonights)girlfriends?"),
            (("A|B*",), "(A|B*)"),
            (("AB|AC",), "A[BC]"),
            (("[BA]?[DC]?[FE]?",), "[AB]?[CD]?[EF]?"),
            (("(ab|cd)?(ef|gh)?", "[BA]?[DC]?[FE]?"),
             "((ab|cd)?(ef|gh)?|[AB]?[CD]?[EF]?)"),
        )
        for wordlist, answer in values:
            self.assertEqual(Regen(wordlist).to_regex(), answer)


if __name__ == "__main__":
    unittest.main()
