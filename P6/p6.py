"""Custom functions for P6 project"""

from ast import literal_eval
import re
import time
import itertools
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.stem import WordNetLemmatizer
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import decomposition
from sklearn import manifold
from sklearn import metrics
from sklearn import preprocessing
from sklearn import pipeline
from sklearn import cluster
from sklearn.model_selection import ParameterGrid


# visualisation des categories dans un dataframe
def category_trees(data):
    """Extrait les catégories et les affiche dans un DataFrame"""

    prod_cat_lists = []
    product_trees = (data["product_category_tree"].apply(lambda x: x.replace(' >> ', '","'))
                     .apply(literal_eval)
                    )
    
    for tree in product_trees:
        prod_cat_lists.append(tree)
    
    df = pd.DataFrame(prod_cat_lists)
    
    
    return df


# création de la variable des catégories de produits
def extract_categories_from_tree(data, level=0):
    """Renvoie uniquement le ou les niveaux de catégories désiré sous forme d'un pandas.Series"""

    df = category_trees(data)

    return df[level]


def load_data(path=None):
    """Chargement du jeu de données"""
    
    if path is None:
        rootpath = "./data/Flipkart/"
        path = rootpath + "flipkart_com-ecommerce_sample_1050.csv"
        data = pd.read_csv(path)
    
    else:
        data = pd.read_csv(path)

    return data


def conf_mat_transform(y_true, y_pred, corresp=None):
    """Construit la matrice de confusion, à partir des étiquettes "vraies" et des étiquettes déterminées 
    par un algorithme de classification/clusterisation ad hoc."""
    
    conf_mat = metrics.confusion_matrix(y_true,y_pred)
        
    if not corresp:
        corresp = np.argmax(conf_mat, axis=0)
    #corresp = [3, 1, 2, 0]
    print ("Correspondance des clusters : ", corresp)
    
    # y_pred_transform = np.apply_along_axis(correspond_fct, 1, y_pred)
    labels = pd.Series(y_true, name="y_true").to_frame()
    labels['y_pred'] = y_pred
    labels['y_pred_transform'] = labels['y_pred'].apply(lambda x : corresp[x]) 
    
    return labels['y_pred_transform']


def word_freq(corpus):
    """Crée le compteur des fréquences des mots du corpus.
    
    """
    # construction du dictionnaire des fréquences
    freq_dist = Counter()
    for text in corpus:
        for token in text.split(' '):
            freq_dist[token] += 1
    
    return freq_dist
    

# Fonction de filtrage des mots (les plus rares et les plus fréquents)
def filter_tokens(corpus, freq_dist, min_tf, max_tf):
    """Cette fonction prend en entrée un ensemble de texte (corpus) et renvoie un 
    nouveau corpus composé des textes purgés trop rares (frequence < min_tf) ou trop
    fréquents (fréquence > max_tf). freq_dist est un Counter() représentant la fréquence
    des tokens composant les textes du corpus."""

    # On retire les tokens tels que word_freq[token] < min_tf ou > max_tf
    new_corpus = [' '.join([token for token in text.split(' ') 
    if (freq_dist[token] > min_tf) and (freq_dist[token] < max_tf)
                            ])
                            for text in corpus
                ]
    return new_corpus
            

def make_docterm_matrix(corpus, idf_transform=True, **kwargs):
    """Crée la matrice document-terme à partir d'un corpus de texte en appliquant la 
    méthode TfidfVectorizer() au corpus. Les arguments sont ceux de la fonction filter_tokens et
    du module TfidfVectorizer() de scikit-learn."""

    freq_dist = word_freq(corpus)
    
    max_tf = kwargs.pop('max_tf', None)
    if max_tf is None:
        max_tf = max(freq_dist.values())
    
    min_tf = kwargs.pop('min_tf', 1)
    corpus = filter_tokens(corpus, freq_dist, min_tf, max_tf)
    
    max_df = kwargs.pop('max_df', 1.0)
    min_df = kwargs.pop('min_df', 1)
    n_gram = kwargs.pop('ngram_range', (1, 1))
    
    tfidfargs = {'max_df':max_df, 'min_df':min_df, 'ngram_range':n_gram}
    
    tfidf = TfidfVectorizer(**tfidfargs)
    values = tfidf.fit_transform(corpus)
    
    return tfidf, values.todense()


def plot_silhouette_analysis(X, data, label_col, savefig=False, path='./images/silh_analysis.png'):
    """Crée et affiche les plots de silhouette pour l'évaluation d'une segmentation
    non-supervisée."""
    
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))
    
    # Compute the silhouette scores for each sample
    sample_silhouette_values = metrics.silhouette_samples(X, data[label_col])

    # Compute the silhouette score for all values
    silhouette_avg = metrics.silhouette_score(X, data[label_col])
    
    y_lower = 10
    for i in range(data[label_col].nunique()):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[data[label_col] == i]
    
        ith_cluster_silhouette_values.sort()
    
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
    
        color = mpl.cm.nipy_spectral(float(i) / data[label_col].nunique())
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )
    
        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    
        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples
    
    ax1.set_title("The silhouette plot for the various clusters.", fontsize=16)
    ax1.set_xlabel("The silhouette coefficient values", size=14)
    ax1.set_ylabel("Cluster label", size=14)
    
    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    
    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.7, -0.5, -0.3, -0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    ax1.set_xticklabels([-0.7, -0.5, -0.3, -0.1, 0, 0.2, 0.4, 0.6, 0.8, 1], size=14)
    
    # 2nd Plot showing the actual clusters formed
    colors = mpl.cm.nipy_spectral(data[label_col].astype(float) / data[label_col].nunique())
    ax2.scatter(
        X[:, 0], X[:, 1], 
        marker=".", 
        s=30,
        lw=0,
        alpha=0.7,
        c=colors,
        edgecolor="k"
    )
    
    
    # Labeling the clusters
    #centers = clusterer.cluster_centers_
    centroids = []
    for t in sorted(data[label_col].unique()):
        cluster_indices = np.where(data[label_col]==t)
        centroids.append(X[cluster_indices].sum(axis=0) / len(cluster_indices[0]))
    
    centroids = np.asarray(centroids)
    
    # Draw white circles at cluster centers
    ax2.scatter(
        centroids[:, 0],
        centroids[:, 1],
        marker="o",
        c="white",
        alpha=1,
        s=200,
        edgecolor="k",
    )
    
    for i, c in enumerate(centroids):
        ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")
    
    ax2.set_title("The visualization of the clustered data.", fontsize=16)
    ax2.set_xlabel("Feature space for the 1st feature", size=14)
    ax2.set_ylabel("Feature space for the 2nd feature", size=14)
    ax2.xaxis.set_tick_params(labelsize=14)
    ax2.yaxis.set_tick_params(labelsize=14)
    

    plt.tight_layout()
    
    if savefig:
        plt.savefig(path, dpi=300)
    
    
    plt.show()


def lda_scorer(corpus, true_labels, ari=False, **kwargs):
    """Crée la matrice document-terme à partir du corpus, calcule un modèle de Dirichlet
    (LatentDirichletAllocation) sur ce corpus pour en extraire les thèmes latents et renvoie, au choix, le score
    de Rand ajusté entre les vrais thèmes et ceux déterminés par le modèle, ou simplement le log-likelihood du modèle
    sur le corpus."""

    # création de la matrice 'bag_of_words'
    _, docterm = make_docterm_matrix(corpus,
                                  **kwargs
                                  )
    
    # Latent Dirichlet Allocation
    n_topics = 7
    doc_topic_prior = kwargs.pop('doc_topic_prior', None)
    topic_word_prior = kwargs.pop("topic_word_prior", None)
    
    lda = decomposition.LatentDirichletAllocation(
        n_components=n_topics, 
        max_iter=10, 
        learning_method='online',
        doc_topic_prior=doc_topic_prior,
        topic_word_prior=topic_word_prior
    )
    
    # Fit
    lda.fit(docterm)
    
    if ari:
        # Evaluation
        # Assignation des catégories à chaque produit: la catégorie retenue est celle pour laquelle la proba est maximale
        X_topics = lda.transform(docterm)
        topics = np.argmax(X_topics, axis=1)
    
        return [metrics.adjusted_rand_score(true_labels, topics)]
    
    else:
        return [lda.score(docterm)]

    
def custom_gridsearch(tokenized_corpus, grid, method='lda', ari=True, labels=None, verbose=1):
    
    results = []
    i = 0
    tinit = time.time()
    grid_length = np.prod([len(v) for v in grid.values()])

    for g in ParameterGrid(grid):
        
        t0 = time.time()
        i += 1
        
        if verbose > 0:
            print("Evaluation du modèle...{}/{}".format(i, grid_length))
            for k, v in g.items():
                print("{} = {}".format(k, v))
            print()
        
        kwargs = g
        
        if method=='lda':
            score = lda_scorer(tokenized_corpus,
                                     labels,
                                     ari=ari,
                                     **kwargs)
            score_name = ["score"]
            
        elif method=='tsne':
            score = lsa_tsne_scorer(tokenized_corpus, labels, **kwargs)
            score_name = ["silhouette_score", "davies_bouldin_score"]
                  
        results.append([*g.values(),
                        *score])
         
        if verbose > 1:
            
            step_duration = time.time() - t0
            global_duration = time.time() - tinit
            avg_per_step = (global_duration + step_duration)/(i + 1)
        
            nb_h = avg_per_step*(grid_length - i) // 3600
            nb_min = (avg_per_step*(grid_length - i) % 3600) // 60
            nb_sec = int((avg_per_step*(grid_length - i) % 3600) % 60)
            print("temps restant estimé: {} h {} min {} s".format(nb_h, nb_min, nb_sec))
            print('-'*40)
            print()
    
    
    cols = list(g.keys())
    for s in score_name:
        cols.append(s)

    results = pd.DataFrame(data=results, columns=cols)

    if method=='lda':
        best_score = results["score"].max()
        index = results[results["score"]==best_score].index[0]
        best_params = results[results["score"]==best_score].transpose().rename(columns={index:"Paramètres"})
        print("Meilleur score: {:.3f}".format(best_score))
        print("Hyperparamètres utilisés pour obtenir le meilleur score: ")
        print(best_params)
        
    return results


def create_lsa_tsne_pipeline(**kwargs):
            
    n_components = kwargs.pop('n_components', 2)
    perplexity = kwargs.pop('perplexity', 30)
    n_iterations = kwargs.pop('iterations', 1000)
    
    kwargs2 = {'n_components':n_components}
    kwargs3 = {'perplexity':perplexity,'n_iter':n_iterations}
    
    lsa_tsne = pipeline.make_pipeline(decomposition.TruncatedSVD(**kwargs2),
                                  preprocessing.Normalizer(copy=False),
                                  manifold.TSNE(**kwargs3))
    
    return lsa_tsne


def lsa_tsne_scorer(corpus, labels, **kwargs):
    

    ngram_range = kwargs.pop('ngram_range', (1,1))
    kwargs1 = {'ngram_range':ngram_range}
    
    n_components = kwargs.pop('n_components', 2)
    perplexity = kwargs.pop('perplexity', 30)
    n_iterations = kwargs.pop('iterations', 1000)
    
    kwargs2 = {'n_components':n_components, 
               'perplexity':perplexity,
               'n_iter':n_iterations}
    
    _, docterm = make_docterm_matrix(corpus, idf_transform=True, **kwargs1)
    lsa_tsne = create_lsa_tsne_pipeline(**kwargs2)
    

    print("Création de la représentation 2D par t-SNE... ")
    print("perplexity = {}, n_iterations = {}, ngram_range = {}".format(kwargs2["perplexity"], 
                                                                        kwargs2["n_iter"],
                                                                        kwargs1["ngram_range"]))
    array_tsne = lsa_tsne.fit_transform(docterm)

    df_tsne = pd.DataFrame(data=array_tsne, columns=["t-SNE 0", "t-SNE 1"])
    silh_score = metrics.silhouette_score(df_tsne, labels)
    db_score = metrics.davies_bouldin_score(df_tsne, labels)
    
    score = [silh_score, db_score]
    return score


def plot_gs_results(results, score, axis_scale="log", plot_param=["alpha"], savefig=False, figname='untitled'):
    """Plots GridSearchCV results.
    
    Draw a plot (one plot per hyperparameter) of mean_train and mean_test scores/errors 
    for each hyperparameter being optimised.
    
    Parameters
    ----------
    cv_results : pandas.DataFrame
        GridSearchCV cross-validation results
    
    score : str or tuple
        Metric(s) used to evaluate the predictions
        
    axis_scale : str, default="log"
        Type of scale to use to draw the plot. Basically 'linear' or 'log' depending on the hyperparameter
    
    plot_param : list, default=["alpha"]
        Hyperparameter against which to plot cross-validated results
        
    savefig : bool, default=False
        Whether or not to save the plot to a .png file.
        
    figname : str, default='untitled'
        Name to give the file if savefig is True.
        
        
    """
    fig = plt.figure(figsize=(5*(len(plot_param)//2 + 1), 4*2))
    
    # Affiche les scores obtenus sur les jeux d'entrainement et de test
    for i, pp in enumerate(plot_param):
        d = results.set_index("{}".format(pp))
        cols = []
        for s in score: 
                cols.append("{}".format(s))
                
        if None in d.index:
            d.rename(index={None: 'None'})
        
        ax = fig.add_subplot(2, len(plot_param)//2 + 1, i+1)
        sns.lineplot(data=d[cols], ci='sd', ax=ax)
        ax.set_xscale(axis_scale)
        ax.set_xlabel(pp, size=12)
        ax.legend(loc='best', frameon=False, fontsize='small')
        
        # We change the fontsize of minor ticks label 
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.tick_params(axis='both', which='minor', labelsize=12)
        
    if savefig:
        plt.savefig(f'./{figname}', dpi=300)
        
    plt.tight_layout()


def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic {}:".format(topic_idx))
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))
        print()


def compute_distorsion(X, topics):
    
    distorsions = []
    
    for t in np.unique(topics):
        cluster_indices = np.where(topics==t)
        centroid = X[cluster_indices].sum(axis=0) / len(cluster_indices[0])
        distorsions.append((np.square(X[cluster_indices] - centroid)).sum(axis=1))
    distorsion = np.sum(list(itertools.chain.from_iterable(distorsions)))
    
    return distorsion


def visualize_tsne(data, X_tsne, label_col, title="K-Means clusterization", savefig=False, path='./images/lsa_kmeans.png'):
    
    X_tsne = pd.DataFrame(data=X_tsne, columns=["t-SNE 0", "t-SNE 1"])
    df = pd.concat([X_tsne, data[[label_col, "product_category_0"]]], axis=1)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(ncols = 2, figsize=(12, 5))

    sns.scatterplot(data=df, x="t-SNE 0", y="t-SNE 1", hue=label_col, palette='viridis', legend=False, ax=ax1)
    #ax1.legend(np.arange(len(np.unique(df.hdbs_label)))-1)
    ax1.set_title(title, fontsize=16)
    ax1.xaxis.set_tick_params(labelsize=14)
    ax1.yaxis.set_tick_params(labelsize=14)

    sns.scatterplot(data=df, x="t-SNE 0", y="t-SNE 1", hue="product_category_0", palette='tab10', ax=ax2)
    ax2.set_title("Original segmentation (level 0)", fontsize=16)
    ax1.xaxis.set_tick_params(labelsize=14)
    # Put the legend out of the figure
    ax2.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    
    for ax in (ax1, ax2):
        ax.set_xlabel('t-SNE 0', size=14)
        ax.set_ylabel('t-SNE 1', size=14)
        ax.xaxis.set_tick_params(labelsize=14)
        ax.yaxis.set_tick_params(labelsize=14)

    plt.tight_layout()

    if savefig:
        plt.savefig(path, dpi=300)

    plt.show()


def hdbs_clustering(X, data, model):

    model.fit(X)
    hdbs_labels = model.labels_
    print("Nombre de clusters trouvés (le bruit compte pour un cluster): {}"
          .format(len(np.unique(hdbs_labels))))
    data["hdbs_label"] = hdbs_labels
    
    return data


def eval_clustering(true_labels, cluster_labels, text="LDA"):
    #Encodage numérique des vraies catégories
    print("ARI entre les étiquettes obtenues par {} et les vraies étiquettes: {:.2f}"
          .format(text, metrics.adjusted_rand_score(true_labels, cluster_labels)))
