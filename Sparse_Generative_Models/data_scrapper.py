import numpy as np
from collections import defaultdict

class DataFilter(object):

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
        vocab_file = open('vocabulary.txt').read()
        for term in vocab_file.split():
            vocabulary[term] = 1.0
        for term in data.split():
            temp_dict[term] += 1.0
        for key in vocabulary.keys():
            vocabulary[key] = vocabulary[key] + temp_dict[key]
        return vocabulary

    def build_vocab(self, data):
        vocabulary = {}
        temp_dict = defaultdict(float)
        vocab_file = open('vocabulary.txt').read()

        for term in vocab_file.split():
            vocabulary[term] = 1.0
        for document in data:
            for term in document.split():
                        temp_dict[term] += 1.0
        for key in vocabulary.keys():
            vocabulary[key] += temp_dict[key]
        return vocabulary

    def category_vocab(self,extractor, categories, type):
        category_dict = {}
        for category in categories:
            print "Extracting " + str(category)
            mycategory_data = extractor.extract_data(type, [category])
            print "Building vocabulary for " + str(category)
            category_dict[category] = self.build_vocab(mycategory_data.data)
        return category_dict

    def laplace_smoothen(self, num_vocab_keys, categories,
                         overall_vocabulary, category_dict):
        log_vocab_prob = {}
        for category in categories:
            log_vocab_prob[category] = {}
            print "Applying Laplace smoothening " + str(category)
            for term in category_dict[category]:
                log_vocab_prob[category][term] = \
                    ((category_dict[category][term] + 1.0)/
                     (overall_vocabulary[term] + num_vocab_keys))
        return log_vocab_prob

    def convert_log_prob(self, categories, log_vocab_prob):
        for category in categories:
            print "Taking log of probabilities for " + str(category)
            for term, prob in log_vocab_prob[category].iteritems():
                log_vocab_prob[category][term] = np.log(prob)
        return log_vocab_prob



