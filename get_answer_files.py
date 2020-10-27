from ranking import Ranking
from information_retrieval import Embedding_retrieval

def get_ranking_nearest(query, ranking, df, doc_k):
    scores = ranking.get_bm25_scores(query)
    sorted_paper_id = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ranking_nearest = [a for a, b in sorted_paper_id
                       if np.all(df.loc[df.paper_id == a,["tag_disease_covid", "after_dec"]].values)
                      ][:doc_k]  # Ranking function nearest
    return ranking_nearest