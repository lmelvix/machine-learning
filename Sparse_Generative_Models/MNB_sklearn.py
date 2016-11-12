from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

class NewsGroup(object):

    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer()
        self.count_vectorizer = CountVectorizer()


    def extract_data(self, subset_type, categories):
        data_set = fetch_20newsgroups(subset=subset_type)
        return data_set

    def vectorize_fitdata(self, data):
        vectors = self.count_vectorizer.fit_transform(data)
        return vectors

    def vectorize_transform(self, data):
        vectors = self.count_vectorizer.transform(data)
        return vectors

    def multinomial_bayesian(self, alpha, train_label, train_vectors, test_vectors, test_label):
        classifier = MultinomialNB(alpha=alpha)
        classifier.fit(train_vectors, train_label)
        prediction = classifier.predict(test_vectors)
        accuracy = metrics.accuracy_score(test_label, prediction)
        return accuracy


def main():

    newsgroup = NewsGroup()
    alpha = 0.01
    train_data = newsgroup.extract_data('train', None)
    train_vector = newsgroup.vectorize_fitdata(train_data.data)
    test_data = newsgroup.extract_data('test', None)
    test_vector = newsgroup.vectorize_transform(test_data.data)

    accuracy = newsgroup.multinomial_bayesian(alpha, train_data.target,
                                           train_vector,  test_vector, test_data.target)

    print str(accuracy*100.0) + "%"

if __name__ == "__main__" :
    main()