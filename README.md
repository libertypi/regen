Module for computing regular expressions from a list of strings and regular
expressions.

The library expands a list of regexes to a finite set of words, then generates a
new regular expression using linear optimization to find the near-shortest
combination. The computed regex should match precisely the same words as the
input.

### Examples:

    >>> from regen import Regen

    >>> wordlist = ['ABC', 'ABD', 'BBC', 'BBD']
    >>> regen = Regen(wordlist)
    >>> regen.to_regex()
    '[AB]B[CD]'

    >>> wordlist = ['[AB]B[CD]', 'XYZ']
    >>> regen = Regen(wordlist)
    >>> regen.to_words()
    ['ABC', 'ABD', 'BBC', 'BBD', 'XYZ']
    >>> regen.to_regex()
    '(XYZ|[AB]B[CD])'
    >>> regen.to_regex(omitOuterParen=True)
    'XYZ|[AB]B[CD]'

### Author: `David Pi`
### Contact: `libertypi@gmail.com`
