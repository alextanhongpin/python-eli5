## CountVectorizer

`CountVectorizer` converts a collection of text documents to a matrix of token counts.


```python
from sklearn.feature_extraction.text import CountVectorizer

vec = CountVectorizer()
vec.fit_transform(["hello world", "I have a dog"]).toarray()
```

Output:
```python
array([[0, 0, 1, 1],
       [1, 1, 0, 0]])
```


If the word does not exists, the matrix returned will be all zero.

```python
print(vec.transform(["hello"]).toarray())
print(vec.transform(["world"]).toarray())
print(vec.transform(["I have a dog"]).toarray())
print(vec.transform(["A random sentence"]).toarray()) # This will return all zeros.
```

Output:
```python
[[0 0 1 0]]
[[0 0 0 1]]
[[1 1 0 0]]
[[0 0 0 0]]
````

Note that the even though our sentences consists of 6 tokens (hello, world, I, have, a, dog), not all words are used as token because single characters are skipped (not because stopwords are filtered).


To know which words are part of the vocab, the `vocabulary_` method provides the index by token.

```python
vec.vocabulary_
```

Output:
```python
{'hello': 2, 'world': 3, 'have': 1, 'dog': 0}
```
