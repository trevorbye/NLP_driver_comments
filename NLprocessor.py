import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer

from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

import LoadCommentEntity
import csv
import matplotlib

#nltk.download("punkt")
#nltk.download("averaged_perceptron_tagger")
#nltk.download("tagsets")
#nltk.download("stopwords")

load_comment_entity_list = []

with open("C:\\Users\\s-tbye\\Desktop\\comments2.txt", "r") as file:
    reader = csv.reader(file)
    for row in reader:
        load_comment_entity_list.append(LoadCommentEntity.CommentEntity(row[0], row[1], row[2:]))

it = iter(load_comment_entity_list)
raw_comment_iterator = iter(load_comment_entity_list)


def build_raw_comment_list(iterator):
    raw_comment_list = []
    comment_exclude_list = []

    comment_exclude_list.append("Prior Split Pickup")
    comment_exclude_list.append("scale")

    for loadCommentEntity in iterator:
        comment = loadCommentEntity.loadComment

        if comment not in comment_exclude_list:
            if not comment.startswith("Red"):
                raw_comment_list.append(comment)

    return raw_comment_list


def build_word_list(comment_list):
    vocabulary_list = []
    exclude_list = []
    exclude_list.append("red")
    exclude_list.append("circl")
    exclude_list.append("prior")
    exclude_list.append("p")
    exclude_list.append("u")
    exclude_list.append("pu")
    exclude_list.append("2")
    exclude_list.append("2nd")

    for comment in comment_list:
        tokens = tokenize_stem_filter(comment)

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
        if token not in stopword_list:
            tokens.append(stemmer.stem(token))

    return tokens
    #freq = nltk.FreqDist(tokens)
    #freq.plot(50)


#build corpus list
raw_comment_list = build_raw_comment_list(raw_comment_iterator)
override_vocabulary_list = build_word_list(raw_comment_list)

#define vectorizer params
vectorizer = TfidfVectorizer(max_df=.10, min_df=5, vocabulary=override_vocabulary_list,
                             use_idf=True, tokenizer=tokenize_stem_filter, ngram_range=(1, 3))

#fit vecorizer to raw driver comment data
tfidf_matrix = vectorizer.fit_transform(raw_comment_list)

#create terms list
terms = vectorizer.get_feature_names()

#define clustering params
k_means = KMeans(n_clusters=8)
k_means.fit_transform(tfidf_matrix)
clusters = k_means.labels_.tolist()

#used to compute relative distances between clusters
#dist = 1 - cosine_similarity(tfidfMatrix)

#loop through cluster centroids and print wordlists
order_centroids = k_means.cluster_centers_.argsort()[:, ::-1]

for cluster in range(8):
    print("Cluster %d wordlist:" % cluster)
    print()

    for ind in order_centroids[cluster, :10]:
        print(terms[ind], end=", ")

    print()
    print()





















