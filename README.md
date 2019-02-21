# Bag-of-Concepts

Re-implementation of ["Bag-of-Concepts: Comprehending Document Representation through Clustering Words in Distributed Representation" (Han Kyul Kim, Hyunjoong Kim, Sunzoon Cho)](https://www.sciencedirect.com/science/article/pii/S0925231217308962)

## Requirements

- gensim >= 3.6.0
- numpy >= 1.15.0
- scikit-learn >= 0.20.1
- scipy >= 1.1.0

## Usage

When input is list of documents (document is represented as str) with white-space tokenizer or your tokenizer

```python
from bag_of_concepts import BOCModel

corpus = ['corpus is list of str format', 'each document is str']
tokenizer = lambda x:x.split()

model = BOCModel(corpus, embedding_method='word2vec', tokenizer=tokenizer)
# or
model = BOCModel(corpus)
boc = model.transform()
```

Or

```python
model = BOCModel()
boc = model.fit_transform(corpus)
```

| Parameter | Type | Default | Help |
| --- | --- | --- | --- |
| input | 'List of str' or 'numpy.ndarray' | None | List of documents. Document is<br>represented with str Or trained word vector representation. |
| n_concepts | int | 100 | Number of concept. |
| min_count | int | 10 | Minumum frequency of word occurrence |
| embedding_dim | int | 100 | Word embedding dimension |
| embedding_method | str | 'word2vec' | Embedding method. Choose from ['word2vec', 'nmf', 'svd'] |
| concept_method | str | 'kmeans' | Concept method. Choose from ['kmeans'] |
| tokenizer | callable | lambda x:x.split() | Return type should be iterable collection of str. |
| idx_to_vocab | list of str | None | Word list. Each word corresponds row of input word vector matrix |
