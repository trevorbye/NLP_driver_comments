import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer

from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

import LoadCommentEntity
import csv
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import jinja2
import mpld3
from TopToolbar import TopToolbar


load_comment_entity_list = []

with open("C:\\Users\\s-tbye\\Desktop\\comments.txt", "r") as file:
    reader = csv.reader(file)
    for row in reader:
        load_comment_entity_list.append(LoadCommentEntity.CommentEntity(row[0], row[1]))

comment_iterator = iter(load_comment_entity_list)
title_iterator = iter(load_comment_entity_list)


def build_raw_comment_list(entity_list):
    comment_list = []
    for entity in entity_list:
        comment_list.append(entity.loadComment)

    return comment_list


def build_title_list(entity_list):
    title_list = []
    for entity in entity_list:
        title_list.append(entity.tankerBase)

    return title_list


def build_vocab_list(terms_list):
    vocabulary_list = []
    exclude_list = ["red", "circl", "prior", "p", "u", "pu", "2", "2nd", "0", "1", "10", "3", "4", "5", "blue", "1st",
                    "c", "l", "tk"]

    for term in terms_list:
        tokens = tokenize_stem_filter(term)

        for token in tokens:
            if token not in vocabulary_list and token not in exclude_list:
                vocabulary_list.append(token)

    return vocabulary_list


def tokenize_stem_filter(text):
    tokens = []
    stopword_list = stopwords.words("english")
    tokenizer = RegexpTokenizer(r'\w+')
    stemmer = PorterStemmer()

    lower_case = text.lower()

    for token in tokenizer.tokenize(lower_case):
        if token not in stopword_list and not token.isdigit():
            tokens.append(stemmer.stem(token))

    return tokens


#build corpus list and milkshed title list for plot output
raw_comment_list = build_raw_comment_list(comment_iterator)
title_list = build_title_list(title_iterator)

#build dummy vectorizer to get intitial wordlist
dummy_vectorizer = TfidfVectorizer(max_df=.15, min_df=5, max_features=200,
                                   use_idf=True, tokenizer=tokenize_stem_filter, ngram_range=(1, 3))

#fit vecorizer to raw driver comment data
temp_matrix = dummy_vectorizer.fit_transform(raw_comment_list)

#create terms list
terms = dummy_vectorizer.get_feature_names()

#build custom vocab list from dummy vectorizer terms list, removing/consolodating duplicates
custom_vocab_list = build_vocab_list(terms)

#trash temp tf-idf objects
del dummy_vectorizer, temp_matrix

#build new vectorizer with custom vocab list
final_vectorizer = TfidfVectorizer(vocabulary=custom_vocab_list, use_idf=True,
                                   tokenizer=tokenize_stem_filter, ngram_range=(1, 3))

final_tfidf_matrix = final_vectorizer.fit_transform(raw_comment_list)

new_terms = final_vectorizer.get_feature_names()

#define clustering params
k_means = KMeans(n_clusters=5)
k_means.fit_transform(final_tfidf_matrix)
clusters = k_means.labels_.tolist()

#used to compute relative distances between clusters
dist = 1 - cosine_similarity(final_tfidf_matrix)

#loop through cluster centroids and print wordlists
order_centroids = k_means.cluster_centers_.argsort()[:, ::-1]

#convert dist matrix to 2d array
#mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
pca = PCA()
pos = pca.fit_transform(dist)

xs, ys = pos[:, 0], pos[:, 1]

#set up colors per clusters
cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e'}

#set up cluster names
cluster_names = {0: "hand, done, finish milk, extra sample, late",
                 1: "call, chart, blue dot, barn, dairy, broken",
                 2: "milk wait, hand, done, milker switch, ice, minute, delay, hose",
                 3: "hose, house, hard, ice, delay, full, hose port",
                 4: "bag, arrive, approve, milk wait, culture sample, farmer, hand, left farm,"}


df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=title_list))
groups = df.groupby("label")

css = """
text.mpld3-text, div.mpld3-tooltip {
  font-family:Arial, Helvetica, sans-serif;
}

g.mpld3-xaxis, g.mpld3-yaxis {
display: none; }

svg.mpld3-figure {
margin-left: -200px;}
"""

#setup plot
fig, ax = plt.subplots(figsize=(17, 9))
ax.margins(0.05)

for name, group in groups:
    points = ax.plot(group.x, group.y, marker='o', linestyle='', ms=12,
            label=cluster_names[name], color=cluster_colors[name],
            mec='none')
    ax.set_aspect('auto')
    labels = [i for i in group.title]

    # set tooltip using points, labels and the already defined 'css'
    tooltip = mpld3.plugins.PointHTMLTooltip(points[0], labels,
                                             voffset=10, hoffset=10, css=css)
    # connect tooltip to fig
    mpld3.plugins.connect(fig, tooltip, TopToolbar())

    # set tick marks as blank
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])

    # set axis as blank
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)


ax.legend(numpoints=1)

mpld3.save_html(fig, "C:\\Users\\s-tbye\\Desktop\\comment_clusters.html")
