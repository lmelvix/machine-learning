import numpy as np
from collections import defaultdict

class DataFilter(object):

    def __init__(self):
        self.vocab_file = open('vocabulary.txt').read()

    def compute_pi(self, labels):
        pi_labels = defaultdict(int)
        total_documents = 0.0
        for label in labels:
            pi_labels[label] += 1.0
            total_documents += 1.0
        for key in pi_labels.keys():
            pi_labels[key] = pi_labels[key]/total_documents
        return pi_labels

    def build_document_vocab(self, data):
        vocabulary = {}
        temp_dict = defaultdict(float)
        for term in self.vocab_file.split():
            vocabulary[term] = 1.0
        for term in data.split():
            temp_dict[term] += 1.0
        for key in vocabulary.keys():
            vocabulary[key] = vocabulary[key] + temp_dict[key]
        return vocabulary

    def category_total_word_count(self, data):
        total_word_count = 0
        vocabulary = {}
        temp_dict = defaultdict(float)
        for term in self.vocab_file.split():
            vocabulary[term] = 1.0
        for document in data:
            for word in document.split():
                temp_dict[word] += 1.0
        for word in vocabulary.keys():
                    vocabulary[word] += temp_dict[word]
                    total_word_count += temp_dict[word]
        return vocabulary, total_word_count

    def num_vocab_words(self):
        vocab_count = 0.0
        for term in self.vocab_file.split():
            vocab_count += 1.0
        return vocab_count

    def category_vocab(self,extractor, categories, type):
        self.category_word_count = {}
        self.total_word_count = {}
        for category in categories:
            print "Extracting " + str(category)
            mycategory_data = extractor.extract_data(type, [category])

            print "Computing total words in " + str(category)
            self.category_word_count, self.total_word_count[category] = \
                self.category_total_word_count(mycategory_data.data)

        return self.category_total_word_count

    def laplace_smoothen(self, num_vocab_keys, categories,total_word_count):
        vocab_prob = {}
        for category in categories:
            vocab_prob[category] = {}
            print "Applying Laplace smoothening " + str(category)
            for term in self.vocab_file.split():
                vocab_prob[category][term] = \
                    ((self.category_word_count[category][term])/
                     (total_word_count[category] + num_vocab_keys))
        return vocab_prob

    def convert_log_prob(self, categories, log_vocab_prob):
        for category in categories:
            print "Taking log of probabilities for " + str(category)
            for term, prob in log_vocab_prob[category].iteritems():
                log_vocab_prob[category][term] = np.log(prob)
        return log_vocab_prob