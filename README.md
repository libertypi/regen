# ReGen: Generate Regular Expressions from Words

ReGen is a Python library for computing regular expressions from a list of strings and regular expressions. It expands a list of regexes to a finite set of words, then generates a new regular expression using linear optimization to find the near-shortest combination. The computed regex should match precisely the same words as the input.

## Examples

### Example 1
Using a simple word list:

```python
from regen import Regen

wordlist = ['ABC', 'ABD', 'BBC', 'BBD']
regen = Regen(wordlist)
result = regen.to_regex()
print(result)  # Output will be '[AB]B[CD]'
```

### Example 2
Using a word list that includes regular expressions:

```python
from regen import Regen

wordlist = ['[AB]B[CD]', 'XYZ']
regen = Regen(wordlist)
words = regen.to_words()
print(words)  # Output will be ['ABC', 'ABD', 'BBC', 'BBD', 'XYZ']

result = regen.to_regex()
print(result)  # Output will be '(XYZ|[AB]B[CD])'

result = regen.to_regex(omitOuterParen=True)
print(result)  # Output will be 'XYZ|[AB]B[CD]'
```

## Author

- **David Pi**
