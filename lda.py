from gensim.models import LdaMulticore, TfidfModel
import pickle
import numpy as np
import gensim
from nltk.stem import WordNetLemmatizer, SnowballStemmer

class LDA:

    def __init__(self, data='texts', lda_parameters=None, tfidf = True):

        self.data = data
        self.tfidf = tfidf
        if lda_parameters is None:
            self.lda_parameters = LDA_Parameters()
        else:
            self.lda_parameters = lda_parameters
        self._get_corpus()
        self.model = LdaMulticore(corpus=self.corpus,
                                  num_topics=self.lda_parameters.num_topics,
                                  id2word=self.token2word,
                                  passes=self.lda_parameters.passes,
                                  workers=self.lda_parameters.workers
                                  )

    def _get_corpus(self):

        stopwords = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now", "al","et"]
        with open('Data/ranking_dict/document_frequencies_text_proc.p', 'rb') as fp:
            document_frequencies = pickle.load(fp)

        word2token = {word: i for i, word in enumerate(document_frequencies.keys())}
        self.word2token = word2token
        self.token2word = {i: word for word, i in word2token.items()}
        del document_frequencies

        with open('Data/ranking_dict/term_frequencies_text_proc.p', 'rb') as fp:
            term_frequencies = pickle.load(fp)
        self.corpus = []
        for _, doc in term_frequencies.items():
            new_doc = []
            for term, freq in doc.items():
                if term in word2token and not term in stopwords and not term.isdigit():
                    new_doc.append((word2token[term], freq))
            self.corpus.append(new_doc)
        print(self.corpus[0])

        if self.tfidf:
            self.tfidf_model = TfidfModel(self.corpus)#, wglobal=lambda tf, d: gensim.models.tfidfmodel.df2idf(tf, d, log_base=20, add=1))
            self.corpus = self.tfidf_model[self.corpus]

    def get_query_vector(self, query, tfidf):
        stemmer = SnowballStemmer("english")
        doc = {}
        for term in set(query.split()):
            term = stemmer.stem(WordNetLemmatizer().lemmatize(term.lower(), pos='v'))
            print(term)
            if term in doc:
                if term in self.token2word:
                    self.token2word[term] += 1
                else:
                    self.token2word[term] = 1
        tf_vec = list(doc.items())
        print(tf_vec)
        if tfidf:
            tf_vec = self.tfidf_model[doc]
            print(tf_vec)
        return self.model[tf_vec]



class LDA_Parameters:

    def __init__(self, num_topics=30, passes=4, workers=4):
        self.num_topics = num_topics
        self.passes = passes
        self.workers = workers
