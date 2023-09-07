# ReGen: Generate Regular Expressions from Words

ReGen is a Python library for computing regular expressions from a list of strings and regular expressions. It expands a list of strings and regexes to a finite set of words, then generates a new regular expression using linear optimization to find the near-shortest result. The computed regex matches precisely the same words as the input.

## Installation

To install ReGen directly from GitHub, run the following command:

```bash
pip install git+https://github.com/libertypi/regen.git
```

## Examples

### Example 1

Convert a list of words into a regular expression:

```python
>>> from regen import Regen

>>> wordlist = ['ABC', 'ABD', 'BBC', 'BBD']
>>> regen = Regen(wordlist)
>>> result = regen.to_regex()
>>> result
'[AB]B[CD]'
```

### Example 2

Convert a list of words and regular expressions into a new regular expression:

```python
>>> from regen import Regen

>>> wordlist = ['[AB]B[CD]', 'XYZ']
>>> regen = Regen(wordlist)
>>> words = regen.to_words()
>>> words
['ABC', 'ABD', 'BBC', 'BBD', 'XYZ']

>>> result = regen.to_regex()
>>> result
'(XYZ|[AB]B[CD])'

>>> result = regen.to_regex(omitOuterParen=True)
>>> result
'XYZ|[AB]B[CD]'
```

### Example 3
regen.py can be used as a command-line utility:
```bash
regen.py --compute cat bat at "fat|boat"
# Output: (bo?|c|f)?at

regen.py --extract "[AB]C[DE]"
# Output:
# ACD
# ACE
# BCD
# BCE

regen.py -f words.txt
# Compute the regex from a word list file, with one word per line.
```

## Author

- **David Pi**