from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer

class NewsGroup(object):

    @staticmethod
    def extract_data(self, subset_type, categories, exclude):
        data_set = fetch_20newsgroups(subset=subset_type,
                                      categories=categories, remove=exclude)
        return data_set

    @staticmethod
    def vectorize_data(data):
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(data)
        return vectors

    @staticmethod
    def multinomial_bayesian(alpha, train_label, train_vectors,
                             test_vectors, test_label):
        classifier = MultinomialNB(alpha=alpha)
        classifier.fit(train_vectors, train_label)
        prediction = classifier.predict(test_vectors)
        f_score = metrics.f1_score(test_label, prediction, average='macro')
        return f_score






newsgroups_train = fetch_20newsgroups(subset='train')
print list(newsgroups_train.target_names)
