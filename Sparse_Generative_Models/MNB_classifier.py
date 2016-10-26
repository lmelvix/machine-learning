import MNB_sklearn as mnb
import data_scrapper as ds


class MNB_Classifier(object):

    def train_mnb_classifier(self):
        extractor = mnb.NewsGroup()
        data_filter = ds.DataFilter()

        print "Extracting complete dataset.."
        overall_train_data = extractor.extract_data('train', None)
        overall_train_categories = overall_train_data.target_names

        print "Computing PI probability for categories.."
        pi_document = data_filter.compute_pi(overall_train_data.target)

        print "Building overall vocabulary"
        overall_vocabulary = data_filter.build_vocab(overall_train_data.data)

        category_dict = data_filter.category_vocab(extractor,
                                                   overall_train_categories, 'train')
        num_vocab_keys = float(len(overall_vocabulary.keys()))

        vocab_prob = data_filter.laplace_smoothen(num_vocab_keys,
                                                  overall_train_categories, overall_vocabulary, category_dict)

        log_vocab_prob = data_filter.convert_log_prob(overall_train_categories,
                                                      vocab_prob)
        print "**Training Complete**\n"
        print log_vocab_prob.keys()


def main():
    mnb_classifier = MNB_Classifier()
    mnb_classifier.train_mnb_classifier()

if __name__ == "__main__" :
    main()