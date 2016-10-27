import MNB_sklearn as mnb
import data_scrapper as ds
import operator
import numpy as np


class MNB_Classifier(object):

    def __init__(self):
        self.extractor = mnb.NewsGroup()
        self.data_filter = ds.DataFilter()

    def train_mnb_classifier(self):

        print "Extracting complete dataset.."
        self.overall_train_data = self.extractor.extract_data('train', None)
        self.overall_train_categories = self.overall_train_data.target_names

        print "Computing PI probability for categories.."
        self.pi_document = self.data_filter.compute_pi(
            self.overall_train_data.target)

        print "Building overall vocabulary"
        self.overall_vocabulary = \
            self.data_filter.build_vocab(self.overall_train_data.data)

        self.category_dict = self.data_filter.category_vocab(self.extractor,
                                                   self.overall_train_categories, 'train')
        self.num_vocab_keys = float(len(self.overall_vocabulary.keys()))

        self.vocab_prob = self.data_filter.laplace_smoothen(self.num_vocab_keys,
                                                       self.overall_train_categories,
                                                       self.overall_vocabulary,
                                                       self.category_dict)

        self.log_vocab_prob = self.data_filter.convert_log_prob(
            self.overall_train_categories,
            self.vocab_prob)
        print "**Training Complete**\n"

    def test_mnb_classifier(self):

        print "Extracting Test data"
        self.test_data = self.extractor.extract_data('test', None)
        self.test_labels = self.test_data.target

        print "**Predicting Test Data**"
        error = 0.0
        total_prediction = 0.0

        for index in range(len(self.test_data.data)):
            test_vocab = self.data_filter.build_document_vocab(self.test_data.data[index])
            test_category_prob = {}
            for category in self.overall_train_categories:
                temp_probability = 1.0
                for term in test_vocab.keys():
                    temp_probability += self.log_vocab_prob[category][term] * (test_vocab[term]-1)
                temp_probability += np.log(self.pi_document[self.overall_train_categories.index(category)])
                test_category_prob[category] = temp_probability
            predict_category = max(test_category_prob.iteritems(), key=operator.itemgetter(1))[0]
            test_category_prob.clear()

            print str(predict_category) + "||" + \
                  str(self.test_data.target_names[self.test_data.target[index]-1])
            if(predict_category != self.test_data.target_names[self.test_data.target[index]]):
                error += 1.0
            total_prediction += 1.0
        print "Error : " + str(error) + "\t Total : " + str(total_prediction)
        print "Error Rate : " + str(error/total_prediction)


def main():
    mnb_classifier = MNB_Classifier()
    mnb_classifier.train_mnb_classifier()
    mnb_classifier.test_mnb_classifier()


if __name__ == "__main__" :
    main()