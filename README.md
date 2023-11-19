# ReGen: Generate Regular Expressions from Words

ReGen is a Python library for computing regular expressions from a list of strings and regular expressions. It expands a list of strings and regexes to a finite set of words, then generates a new regular expression using linear optimization to find the near-shortest result. The computed regex matches precisely the same words as the input.

## Examples

### Alphanumeric Code

*Two-Letter Abbreviations of the 50 US States*

`AL, AK, AZ, AR, CA, CZ, CO, CT, DE, DC, FL, GA, GU, HI, ID, IL, IN, IA, KS, KY, LA, ME, MD, MA, MI, MN, MS, MO, MT, NE, NV, NH, NJ, NM, NY, NC, ND, OH, OK, OR, PA, PR, RI, SC, SD, TN, TX, UT, VT, VI, VA, WA, WV, WI, WY`

**Result:**

`A[KLRZ]|C[AOTZ]|DC|DE|FL|GU|HI|I[ADLN]|KS|KY|M[ADEINOST]|N[CDEHJMVY]|O[HKR]|PR|RI|SC|SD|TN|TX|UT|V[AIT]|W[AIVY]|[GLP]A`

### Natural Language

*"Imagined Communities" by Benedict Anderson, first paragraph of Introduction, with non-word characters removed. 136 words*

```
perhaps without being much noticed yet a fundamental transformation in the history of marxism and marxist
movements is upon us its most visible signs are the recent wars between vietnam cambodia and china these wars
are of world historical importance because they are the first to occur between regimes whose independence and
revolutionary credentials are undeniable and because none of the belligerents has made more than the most
perfunctory attempts to justify the bloodshed in terms of a recognizable marxist theoretical perspective while
it was still just possible to interpret the sino soviet border clashes of and the soviet military
interventions in germany hungary czechoslovakia and afghanistan in terms of according to taste social
imperialism defending socialism etc no one i imagine seriously believes that such vocabularies have much
bearing on what has occurred in indochina
```

**Result:**

```
((believ|clash|regim|vocabulari)e|(belliger|movem)ent|attempt|credential|ha|i|intervention|it|perhap|sign|term
|u|war?)s|((fir|ju|marxi|mo)s|[tw]ha|i|interpre|recen|sovie|withou|ye)t|((his|perfunc)tor|(hung|milit|revoluti
on)ar|german|justif|seriousl|the)y|((imper|soc)ial|marx)ism|((possi|recogniza|undenia|visi)bl|ar|becaus|hav|i(
mporta|ndepende)nc|imagin|mad|mor|n?on|perspectiv|tast|th|thes|whil|whos)e|(accord|be|bear|defend)ing|(afghani
sta|betwee|i|o|tha|transformatio|upo)n|(an|bloodshe|notice|worl)d|(fundament|historic|soci|theoretic)al|(indo)
?china|[ms]uch|a|border|c(ambod|zechoslovak)ia|etc|i|no|occur(red)?|of|sino|still|to|vietnam
```

## Installation

To install ReGen directly from GitHub, run the following command:

```bash
pip install git+https://github.com/libertypi/regen.git
```

## Usage

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

Extract a list of regular expressions to a word list:

```python
>>> from regen import Regen

>>> wordlist = ['[AB]B[CD]', 'XYZ']
>>> regen = Regen(wordlist)
>>> words = regen.to_words()
>>> words
['ABC', 'ABD', 'BBC', 'BBD', 'XYZ']
```

Then convert it into a new regular expression:

```python
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
regen.py --compute "cat" "bat" "at" "fat|boat"
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
