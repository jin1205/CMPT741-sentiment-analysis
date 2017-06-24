import os
import numpy as np
from tempfile import TemporaryFile

BASE_DIR = '.'
GLOVE_DIR = BASE_DIR + '/glove.840B/'

embeddings_index = {}

f = open(os.path.join(GLOVE_DIR, 'glove.840B.300d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

embedding_matrix = 0.1 * np.random.randn(14099, 300)

vocab = open(os.path.join(BASE_DIR, 'vocab.dat'))

j = 0
for line in vocab:
	ls = line.split(' ', 1)
	i = int(ls[0])
	word = ls[1]
	embedding_vector = embeddings_index.get(word.rstrip())
	if embedding_vector is not None:
		embedding_matrix[i-1] = embedding_vector
		j = j + 1

np.savetxt('wordvector_300_840B.txt', embedding_matrix)
