import matplotlib.pyplot as plt
import numpy as np
import csv
import configuration as config
import random
import heapq





################################################################
class TopN(object):
	"""Maintains the top n elements of an incrementally provided set."""

	def __init__(self, n):
		self._n = n
		self._data = []

	def size(self):
		assert self._data is not None
		return len(self._data)

	def push(self, x):
		"""Pushes a new element."""
		assert self._data is not None
		if len(self._data) < self._n:
			heapq.heappush(self._data, x)
		else:
			heapq.heappushpop(self._data, x)

	def extract(self, sort=False):
		"""Extracts all elements from the TopN. This is a destructive operation.
		The only method that can be called immediately after extract() is reset().
		Args:
			sort: Whether to return the elements in descending sorted order.
		Returns:
			A list of data; the top n elements provided to the set.
		"""
		assert self._data is not None
		data = self._data
		self._data = None
		if sort:
			data.sort(reverse=True)
		return data

	def reset(self):
		"""Returns the TopN to an empty state."""
		self._data = []

################################################################

class OutputSentence(object):
	"""Represents a complete or partial caption."""

	def __init__(self, sentence, state, logprob, score, metadata=None):
		"""Initializes the Caption.
		Args:
			sentence: List of word ids in the caption.
			state: Model state after generating the previous word.
			logprob: Log-probability of the caption.
			score: Score of the caption.
			metadata: Optional metadata associated with the partial sentence. If not
				None, a list of strings with the same length as 'sentence'.
		"""
		self.sentence = sentence
		self.state = state
		self.logprob = logprob
		self.score = score
		self.metadata = metadata

	def __cmp__(self, other):
		"""Compares Captions by score."""
		assert isinstance(other, OutputSentence)
		if self.score == other.score:
			return 0
		elif self.score < other.score:
			return -1
		else:
			return 1
	
	# For Python 3 compatibility (__cmp__ is deprecated).
	def __lt__(self, other):
		assert isinstance(other, OutputSentence)
		return self.score < other.score
	
	# Also for Python 3 compatibility.
	def __eq__(self, other):
		assert isinstance(other, OutputSentence)
		return self.score == other.score

################################################################

def sampleFromDistribution(vals):
		p = random.random()
		s=0.0
		for i,v in enumerate(vals):
				s+=v
				if s>=p:
						return i
		return len(vals)-1


def getCMUDictData(fpath="./data/cmudict-0.7b"):
	print "--- Loading CMU data"
	data = open(fpath,"r").readlines()
	data = data[126:]
	data = [row.split(' ')[0] for row in data]
	data = [row.lower() for row in data]
	print "length of cmu data ",len(data)
	print "A couple of samples..."
	print data[0]
	print data[1]
	print data[2]
	print "------------"
	return data


if __name__ == "__main__":
	getCMUDictData()
