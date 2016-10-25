from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer

class NewsGroup(object):

    def __init__(self):
            self.vectorizer = TfidfVectorizer()

    def extract_data(self, subset_type):
        data_set = fetch_20newsgroups(subset=subset_type)
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

    def calculate_probability(self, train_):


def main():
    newsgroup = NewsGroup()
    alpha = 0.1

    train_data = newsgroup.extract_data('train')
    train_vector = newsgroup.vectorize_fitdata(train_data.data)
    test_data = newsgroup.extract_data('test')
    test_vector = newsgroup.vectorize_transform(test_data.data)

    error = newsgroup.multinomial_bayesian(alpha, train_data.target,
                                           train_vector,  test_vector, test_data.target)

    print error

if __name__ == "__main__" :
    main()