import numpy as np
import scipy as sp

class BOCModel:
    """
    :param input: 'List of list of str' or 'numpy.ndarray'
        List of documents. Document is represented with list of str
        Or trained word vector representation.
    :param n_concepts: int
        Number of concept.
    :param min_count: int
        Minumum frequency of word occurrence
    :param embedding_dim: int
        Word embedding dimension
    :param embedding_method: str
        Embedding method. Choose from ['word2vec', 'svd']
    :param concept_method: str
        Concept method. Choose from ['kmeans']
    :param tokenizer: callable
        Return type should be iterable collection of str.
        Default is lambda x:x.split()
    :param idx_to_vocab: list of str
        Word list. Each word corresponds row of input word vector matrix

    Attributes
    ----------
    wv : numpy.ndarray
        Word vector representation
    idx_to_vocab : list of str
        Word list. Each word corresponds row of input word vector matrix
    idx_to_concept : numpy.ndarray
        Word to concept index
    idx_to_concept_weight : numpy.ndarray
        Word to concept weight
    """

    def __init__(self, input=None, n_concepts=100, min_count=10, embedding_dim=100,
        embedding_method='word2vec', concept_method='kmeans', tokenizer=None,
        idx_to_vocab=None, verbose=True):

        if not embedding_method in ['word2vec', 'svd']:
            raise ValueError("embedding_method should be ['word2vec', 'svd']")
        if not concept_method in ['kmeans']:
            raise ValueError("concept_method should be ['kmeans']")
        if min_count < 1 and isinstance(min_count, int):
            raise ValueError('min_count should be positive integer')

        self.n_concepts = n_concepts
        self.min_count = min_count
        self.embedding_dim = embedding_dim
        self.embedding_method = embedding_method
        self.concept_method = concept_method
        self.verbose = True

        if tokenizer is None:
            tokenizer = lambda x:x.split()
        if not callable(tokenizer):
            raise ValueError('tokenizer should be callable')
        self.tokenizer = tokenizer

        self.wv = None
        self.idx_to_concept = None
        self.idx_to_concept_weight = None

        if isinstance(input, np.ndarray):
            if idx_to_vocab is None:
                raise ValueError('idx_to_vocab should be inserted '\
                                 'when input is word vector')
            if len(idx_to_vocab) != input.shape[0]:
                a = len(idx_to_vocab)
                b = input.shape[0]
                raise ValueError('Length of idx_to_vocab is different '\
                                 'with input matrix %d != %d' % (a, b))
            self.idx_to_vocab = idx_to_vocab
            self._train_concepts(input)
        elif not isinstance(input, None):
            self.idx_to_vocab = None
            self.fit_transform(input)
        else:
            self.idx_to_vocab = None

    def fit_transform(self, corpus):
        if isinstance(corpus, np.ndarray):
            raise ValueError('Input corpus should not be word vector')

        if ((self.wv is None)
             or (self.idx_to_concept is None)
             or (self.idx_to_vocab is None)):
            self.fit(corpus)
        return self.transform(corpus)

    def fit(self, corpus):
        self._train_word_embedding(corpus)
        self._train_concept(self.wv)

    def _train_word_embedding(self, corpus):
        if self.embedding_method == 'word2vec':
            self.wv, self.idx_to_vocab = train_wv_by_word2vec(
                corpus, self.min_count, self.embedding_dim)
        elif self.embedding_method == 'svd':
            self._bow, self.idx_to_vocab = corpus_to_bow(
                corpus, self.tokenizer, self.idx_to_vocab, self.min_count)
            self.wv = train_wv_by_svd(bow, self.embedding_dim)
        else:
            raise ValueError("embedding_method should be ['word2vec', 'svd']")

    def _train_concept(self, wv):
        idx_to_c, idx_to_cw = train_concept_by_kmeans(self.wv, self.n_concepts)
        self.idx_to_concept = idx_to_c
        self.idx_to_concept_weight = idx_to_cw
        raise NotImplemented

    def transform(self, corpus_or_bow, remain_bow=False):
        # use only trained vocabulary
        if not sp.sparse.issparse(corpus_or_bow):
            self._bow = corpus_to_bow(corpus, self.tokenizer,
                self.idx_to_vocab, min_count=-1)
        else:
            self._bow = corpus_or_bow

        # concept transformation
        boc = bow_to_boc(self._bow, self.idx_to_concept,
            self.idx_to_concept_weight)

        if not remain_bow and hasattr(self, '_bow'):
            del self._bow

        return boc

def corpus_to_bow(corpus, tokenizer, idx_to_vocab=None, min_count=-1):
    raise NotImplemented

def train_wv_by_word2vec(corpus, min_count, embedding_dim):
    raise NotImplemented

def train_wv_by_svd(bow, embedding_dim):
    raise NotImplemented

def train_concept_by_kmeans(wv, n_concepts):
    raise NotImplemented

def bow_to_boc(bow, idx_to_concept, idx_to_concept_weight):
    raise NotImplemented
