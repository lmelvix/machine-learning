import MNB_sklearn as mnb
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

def main():

    extractor = mnb.NewsGroup()
    data_filter = DataFilter()

    print "Extracting complete dataset.."
    overall_train_data = extractor.extract_data('train', None)
    overall_train_categories = overall_train_data.target_names

    print "Computing PI probability for categories.."
    pi_document = data_filter.compute_pi(overall_train_data.target)

    print "Building overall vocabulary"
    overall_vocabulary = data_filter.build_vocab(overall_train_data.data)

    category_dict = {}
    for category in overall_train_categories:
        print "Extracting " + str(category)
        mycategory_data = extractor.extract_data('train', [category])
        print "Building vocabulary for " + str(category)
        category_dict[category] = data_filter.build_vocab(mycategory_data.data)

    log_vocab_prob = {}
    for category in overall_train_categories:
        log_vocab_prob[category] = {}
        print "Building log of word probabilities for " + str(category)
        for term in category_dict[category]:
            log_vocab_prob[category][term] = \
                np.log(float(category_dict[category][term] + 1.0)
                       /float(overall_vocabulary[term] + float(len(overall_vocabulary.keys()))))

    print "COMPLETED !!"
    print log_vocab_prob['alt.atheism']['crawford']

if __name__ == "__main__" :
    main()