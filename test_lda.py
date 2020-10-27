from lda import *
from gensim.similarities import MatrixSimilarity


print('Calculating LDA model...')
lda_model = LDA()

for idx, topic in lda_model.model.print_topics(-1):
    print("Topic: {} \nWords: {}".format(idx, topic))
    print("\n")


index = MatrixSimilarity(lda_model.model[lda_model.corpus])
query = 'Range of incubation periods for the disease in humans (and how this varies across age and health status) and how long individuals are contagious, even after recovery.'
q_vec = lda_model.get_query_vector(query)

sims = index[q_vec]
sims = sorted(enumerate(sims), key=lambda item: -item[1])
print(sims[:10])