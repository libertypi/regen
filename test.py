#!/usr/bin/env python3

import unittest

from regen import Regen


class TestRegen(unittest.TestCase):

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
            'A[B3-5]C': ['A3C', 'A4C', 'A5C', 'ABC'],
            'AB|C()D': ['AB', 'CD'],
            '(A(B|C(D|E)*)*|F)': ['A(B|C[DE]*)*', 'F'],
            'A(\\B*\\C|\\D)?': ['A', 'A\\B*\\C', 'A\\D'],
            '(ABC)*': ['(ABC)*'],
            '((ABC)?)*': ['((ABC)?)*'],
            '(A|B|C|D){3}': ['[A-D]{3}'],
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


    def test_range_expansion(self):
        """Parser expands character ranges into individual characters."""
        values = {
            # Basic range
            '[a-e]': list('abcde'),
            # Digits
            '[0-9]': list('0123456789'),
            # Range mixed with literal chars
            '[a-cxy]': ['a', 'b', 'c', 'x', 'y'],
            # Multiple ranges
            '[a-c0-2]': ['0', '1', '2', 'a', 'b', 'c'],
            # Single-char range
            '[a-a]': ['a'],
            # Range with prefix/suffix
            'X[a-d]Y': ['XaY', 'XbY', 'XcY', 'XdY'],
            # Range with ? suffix expands
            '[a-c]?': ['', 'a', 'b', 'c'],
            # Range with * suffix stays opaque
            '[a-c]*': ['[a-c]*'],
            # Range with {n} suffix stays opaque
            '[a-c]{3}': ['[a-c]{3}'],
            # Dash at end is literal, not range
            '[a-c-]': ['-', 'a', 'b', 'c'],
            # Bracket at start is literal
            '[]a-c]': [']', 'a', 'b', 'c'],
            # Full lowercase alphabet
            '[a-z]': list('abcdefghijklmnopqrstuvwxyz'),
        }
        for regex, answer in values.items():
            result = Regen([regex]).to_words()
            self.assertEqual(result, sorted(answer), msg=f"{regex}: {result}")

    def test_range_expansion_errors(self):
        """Parser raises on invalid ranges."""
        with self.assertRaises(ValueError):
            Regen(['[z-a]']).to_words()

    def test_range_collapsing(self):
        """Optimizer collapses consecutive characters into ranges."""
        values = (
            # 3 consecutive: no range (same length as listing)
            (list('ABC'), '[ABC]'),
            # 4+ consecutive -> range
            (list('ABCDEF'), '[A-F]'),
            # Full digits
            (list('0123456789'), '[0-9]'),
            # Mixed consecutive and non-consecutive
            (list('ABCXYZ'), '[ABCXYZ]'),
            # 2 consecutive: no range
            (list('AB'), '[AB]'),
            # Non-consecutive scattered chars
            (list('ace'), '[ace]'),
            # Single char: no brackets
            (['A'], 'A'),
            # Multiple disjoint ranges
            (list('abcdwxyz') + list('0123'), '[0-3a-dw-z]'),
            # Full lowercase alphabet
            (list('abcdefghijklmnopqrstuvwxyz'), '[a-z]'),
        )
        for wordlist, answer in values:
            result = Regen(wordlist).to_regex()
            self.assertEqual(result, answer, msg=f"{wordlist}: {result}")

    def test_range_collapsing_special_chars(self):
        """Optimizer handles ] and - correctly with range collapsing."""
        values = {
            # Range + dash (via regex input)
            '[ABCD-]': '[A-D-]',
            # Range + bracket (via regex input)
            '[]ABCD]': '[]A-D]',
            # Range + bracket + dash
            '[]ABCD-]': '[]A-D-]',
        }
        for regex, answer in values.items():
            result = Regen([regex]).to_regex()
            self.assertEqual(result, answer, msg=f"{regex}: {result}")

    def test_range_roundtrip(self):
        """Ranges survive parse -> optimize -> parse roundtrip."""
        values = (
            '[a-z]',
            '[A-Za-z0-9]',
            '[a-dx-z]',
            '[0-9]',
            'A[a-f]B',
            '[A-D][0-3]',
        )
        for regex in values:
            regen = Regen([regex])
            words = regen.to_words()
            result = Regen(words).to_regex()
            roundtrip_words = Regen([result]).to_words()
            self.assertEqual(
                words, roundtrip_words,
                msg=f"{regex} -> {result}: words differ"
            )

    def test_range_verify(self):
        """Verify that optimized regex with ranges matches original words."""
        test_inputs = (
            list('ABCDEF'),
            list('0123456789'),
            list('abcxyz'),
            ['A0', 'A1', 'A2', 'A3', 'B0', 'B1', 'B2', 'B3'],
        )
        for wordlist in test_inputs:
            regen = Regen(wordlist)
            regen._verify()


if __name__ == "__main__":
    unittest.main()
