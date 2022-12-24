import gensim
import numpy as np
from scipy import sparse


class MyCorpus():
    """A memory-friendly iterator that yields documents as TaggedDocument objects, i.e tokens associated with index of document."""
    
    def __init__(self, data, vocab, tokens_only=False):
        self.data = data
        self.vocab = vocab
        self.tokens_only = tokens_only
    
    def __iter__(self):
        if isinstance(self.data, sparse.csr_matrix):
            for i, x in enumerate(self.data):
                tokens = list(self.vocab[x.indices])
                yield gensim.models.doc2vec.TaggedDocument(tokens, [i])
        else:
            if not self.tokens_only:
                for i, x in enumerate(self.data):
                    tokens = list(self.vocab[np.flatnonzero(x)])
                    yield gensim.models.doc2vec.TaggedDocument(tokens, [i])
            else:
                for i, x in enumerate(self.data):
                    tokens = list(self.vocab[np.flatnonzero(x)])
                    yield tokens