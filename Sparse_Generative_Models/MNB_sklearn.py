from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer

class NewsGroup(object):

    def __init__(self):
            self.vectorizer = TfidfVectorizer()

    def extract_data(self, subset_type, categories):
        data_set = fetch_20newsgroups(subset=subset_type,
                                      categories=categories,
                                      remove=('headers', 'footers', 'quotes'))
        return data_set

    def vectorize_fitdata(self, data):
        vectors = self.vectorizer.fit_transform(data)
        return vectors

    def vectorize_transform(self, data):
        vectors = self.vectorizer.transform(data)
        return vectors

    def multinomial_bayesian(self, alpha, train_label, train_vectors, test_vectors, test_label):
        classifier = MultinomialNB(alpha=alpha)
        classifier.fit(train_vectors, train_label)
        prediction = classifier.predict(test_vectors)
        f_score = metrics.f1_score(test_label, prediction, average='macro')
        return f_score


def main():

    newsgroup = NewsGroup()
    alpha = 0.01
    train_data = newsgroup.extract_data('train')
    train_vector = newsgroup.vectorize_fitdata(train_data.data)
    test_data = newsgroup.extract_data('test')
    test_vector = newsgroup.vectorize_transform(test_data.data)

    f_score = newsgroup.multinomial_bayesian(alpha, train_data.target,
                                           train_vector,  test_vector, test_data.target)
    print f_score

if __name__ == "__main__" :
    main()