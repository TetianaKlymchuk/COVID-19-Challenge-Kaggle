import numpy as np
import pandas as pd
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
assert gensim.models.doc2vec.FAST_VERSION > -1, "This will be painfully slow otherwise"
from scipy.spatial import distance
import re
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from sklearn.cluster import KMeans
from sklearn import metrics
import scipy.spatial.distance
import matplotlib.pyplot as plt


class Embedding_doc2vec():
    """
    Class that handles document embedding doc2vec
    """

    def __init__(self, df=None, path=None, embedding_dim=50, epochs=2,  min_count=2):
        """
        :param df:  dataframe with columns ['paper_id', 'title', 'abstract','text']
        :param path: path to pretrained embedding.
        :param vector_size: vectors dimension.
        :param epoch: number of iterations (epochs) over the corpus.
        :param min_count: minimum frequency of a word in the corpus to consider it.
        """
        if path is None and df is None:
            raise Exception('You must provide either a df or a path')
        if df is None:
            self.load(path)
        else:
            model = self._get_embedding(df, vector_size=embedding_dim, epochs=epochs,  min_count=min_count)
            self.model = model
            
            
    def preprocess_quotations(quotation, stopwords = False):
        """
        Method to tokenize each text without stopwords.
        :param quotation: text <string>
        :param stopwords:  use or not stopwords <bool>
        :return: text vector <np.array>
        """
        for i, quotation in enumerate(quotation):
            if stopwords:
                stop_words = set(stopwords.words('english')) 
                for term in ['et', 'al', 'also', 'fig']:
                    stop_words.add(term)
                word_tokens = gensim.utils.simple_preprocess(quotation) 
                yield [w for w in word_tokens if not w in stop_words]
            else:
                yield gensim.utils.simple_preprocess(quotation)
       
    
    def _get_embedding(self, df, vector_size=50, epochs=2, min_count=2):
        """
        Method to obtain gensim Doc2Vec model trained on all texts.
        :param df: inicial df <DataFrame>
        :param vector_size: vectors dimension.
        :param epoch: number of iterations (epochs) over the corpus.
        :param min_count: minimum frequency of a word in the corpus to consider it.
        :return: gensim Doc2Vec model trained on all texts.
        """
        # preprocessing texts
        all_texts = df.text
        all_corpus = [(gensim
                       .models
                       .doc2vec
                       .TaggedDocument(self.preprocess_quotations(quotation, stopwords = True), [i]))
                      for i, quotation in enumerate(all_texts)]
              
        model = (gensim
                 .models
                 .doc2vec
                 .Doc2Vec(vector_size=vector_size, min_count=min_count, epochs=epochs))
        model.build_vocab(all_corpus)
        model.train(all_corpus, total_examples=model.corpus_count, epochs=model.epochs)

        return model

    
    def get_embedding(self, paper_id, df):
        """
        Method to obtain the embedding vector of a document.
        :param df: inicial df <DataFrame>
        :param paper_id: id of a document <string>
        :return: text vector <np.array>
        """
        idx = df[df.paper_id==paper_id].index.values[0]
        
        return self.model.docvecs[idx]
        
    
    
    def get_dataframe(self, df):
        """
        Method to obtain ta dataframe with the embedding vector for each paper_id.
        :param df: inicial df <DataFrame>
        :return: a dataframe <DataFrame>
        """
        """all_vectors = np.array([(self
                                 .model
                                 .docvecs[ind_corpus]) for ind_corpus in range(len(df))])"""
        paper_id = df.paper_id.values
        dic_doc2vec = {}

        for ind_corpus in range(len(df)):
            dic_doc2vec[paper_id[ind_corpus]] = (self.model.docvecs[ind_corpus])

        df_doc2vec = pd.DataFrame.from_dict(dic_doc2vec, orient='index')
        df_doc2vec.reset_index(inplace=True)

        return df_doc2vec
    
    
    def get_distances(self, df, paper_id):
        """
        Method to obtain all distances of one given text to all texts.
        :param df: inicial df <DataFrame>
        :param paper_id: id of a given vector <string>
        :return: number vector <np.array>
        """
        df_doc2vec = self.get_dataframe(df)
        paper_vector = self.get_embedding(paper_id, df)
        distances = (df_doc2vec
                     .iloc[:,1:]
                     .apply(lambda x: (scipy
                                       .spatial
                                       .distance
                                       .cosine(x, paper_vector)), axis=1))
        
        return distances
    
    
    def get_nearest(self, df, paper_id, k):
        """
        Method to obtain all distances of one given text to all texts.
        :param df: inicial df <DataFrame>
        :param paper_id: id of a given vector <string>
        :param k: number of nearest vectors <integer>
        :return: number vector <np.array>
        """
        distances = self.get_distances(df, paper_id)
        k_nearest = distances[distances != 0].nsmallest(n=k).index
        
        return [df.iloc[elem].paper_id for elem in k_nearest]
   

    def plot_distances(self, df, paper_id, threshold=0.3):
        """
        Method to obtain ta dataframe with the embedding vector for each paper_id.
        :param df: inicial df <DataFrame>
        :param paper_id: id of a given vector <string>
        :param threshold: a maximal distance to the given vector <integer>
        :return: a scatter plot 
        """
        distances = self.get_distances(df, paper_id)
        y = distances[distances<threshold]
        x = np.linspace(0, len(y), num=len(y))

        plt.figure(figsize=(8, 4), dpi=80)
        plt.scatter(x, y, alpha=0.5)
        plt.title("Number of articles under the treshould")
        plt.xlabel("Index")
        plt.ylabel("Distance")
        plt.show()
   
    
    def save(self, path = 'doc2vecmodel.mod'):
        self.model.save(path)

        
    def load(self, path = 'Data/doc2vecmodel.mod'):
        self.model = Doc2Vec.load(path)

        
    def remove_punct(self,text):
        """
        An additional function to remove punctuation from the query
        :param text: given text/abstract/paragraph <string>
        :return: string
        """
        new_words = []
        for word in text:
            w = re.sub(r'[^\w\s]','',word) #remove everything except words and space                        
            w = re.sub(r'\_','',w) #to remove underscore as well
            new_words.append(w)
        return new_words
    
    
    def filtered_query(self, query):
        """
        Method to obtain cleaned query from a given query.
        :param query: given text/abstract/paragraph <string>
        :return: list of strings
        """
        query = " ".join(self.remove_punct(query.split()))
        stop_words = set(stopwords.words('english'))
        for term in ['et', 'al', 'also', 'fig']:
            stop_words.add(term)
        word_tokens = word_tokenize(query) 

        return [w for w in word_tokens if not w in stop_words] 
        
    
    def most_similar(self, df, k, query):
        """
        Method to get the k nearest neighbors to the document.
        :param word: word to find nearest neighbors.
        :param k: number of neighbors to return
        :return: list of (word, similarity)
        """
        ivec = self.model.infer_vector(doc_words = self.filtered_query(query), steps=20, alpha=0.025)
        most_similar  = self.model.docvecs.most_similar(positive=[ivec], topn=k)
        idx = [idx for (idx, dist) in most_similar]
        return [df.iloc[elem].paper_id for elem in idx]

    
    def get_clusters(self, df, k=20):
        """
        Method to obtain K-Means clustering of doc2vec representations
        :param df: inicial df <DataFrame>
        :param k: number of clusters <integer>
        :return: np.array
        """
        df_doc2vec = self.get_dataframe(df)
        all_vectors = np.array([self.model.docvecs[ind_corpus] for ind_corpus in range(len(df))])
        kmeans = KMeans(n_clusters=k, n_jobs=4, verbose=10)
        y_pred = kmeans.fit_predict(all_vectors)
        
        return y_pred
    
    
    def get_cluster_for_text(self, df, clusters, paper_id):
        """
        Method to obtain a cluster of a given paper_id
        :param df: inicial df <DataFrame>
        :param clusters: vector of clusters distributions from self.get_clusters <np.array>
        :param k: number of clusters <integer>
        :return: integer
        """
        idx = df.loc[df.paper_id==paper_id].index.values[0]
        
        return clusters[idx]
    
    
    def common_cluster(self, df, clusters, vector):
        """
        Method to get paper_ids that are in the same cluster that have been obtained by different methods
        :param df: inicial df <DataFrame>
        :param clusters: vector of clusters distributions from self.get_clusters <np.array>
        :param vector1: number of clusters <integer>
        :return: list of strings
        """
        vector_aux = [df[df.paper_id==paper_id].index.values[0] for paper_id in vector]
        cl_vector = [clusters[idx] for idx in vector_aux]
        res = [np.argwhere(i[0]==cl_vector) for i in np.array(np.unique(cl_vector, return_counts=True)).T if i[1]>=2]
        if len(res)>0:
            return [[vector[unit[0]] for unit in elem] for elem in res]
        
        return []
           
        
    def get_one_cluster(self, df, clusters, num):
        """
        Method to get paper_ids that are in the same cluster that have been obtained by different methods
        :param df: inicial df <DataFrame>
        :param clusters: vector of clusters distributions from self.get_clusters <np.array>
        :param num: number of cluster <integer>
        :return: list of strings
        """
        return [df.iloc[elem].paper_id for elem in np.where(clusters==num)[0]]

    def get_cluster_dict(self, df, clusters, vector):
        """
        Returns a dictionary cluster <int> -> number of documents in vector that belong to that cluster <int>.
        :param df: initial df <DataFrame>
        :param clusters: vector of clusters distributions from self.get_clusters <np.array>
        :param vector: list of paper ids.
        :return: dictionary cluster -> number of documents from vector
        """
        comm_cluster = self.common_cluster(df, clusters, vector=vector)
        d = {}
        for el in comm_cluster:
            key = self.get_cluster_for_text(df, clusters, el[0])
            d[key] = el
        return d
        
        


   