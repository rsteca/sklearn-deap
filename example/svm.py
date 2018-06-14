import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, train_test_split, StratifiedShuffleSplit
from evolutionary_search import EvolutionaryAlgorithmSearchCV

def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    plt.draw()
    plt.pause(0.005)
    return out

random.seed(1)

iris = datasets.load_iris()
# only select the first two features for ease of plotting: Sepal length; Sepal width
x_train, x_test, y_train, y_test = train_test_split(
    iris.data[:, :2], iris.target.reshape(150, 1))


if __name__ == '__main__':

    generic_args = {'scoring': "accuracy",
                    # cross validation method
                    'cv': StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42),
                    'verbose': True,
                    # genetic algorithm params are quite straightforward to pick though
                    # if the output looks rubbish do some googling and tweak
                    'population_size': 300,
                    'gene_mutation_prob': 0.1,
                    'gene_crossover_prob': 0.5,
                    'tournament_size': 4,
                    'generations_number': 100,
                    'n_jobs': 6}  # parallel processes

    # we create an instance of SVM and fit out data. We do not scale our
    # data since we want to plot the support vectors
    models = (EvolutionaryAlgorithmSearchCV(estimator=SVC(),
                                            params={"kernel": ["rbf"],
                                                    "C":      np.logspace(-9, 9, num=10000, base=10),
                                                    "gamma":  np.logspace(-9, 9, num=10000, base=10)},
                                            **generic_args),
              EvolutionaryAlgorithmSearchCV(estimator=SVC(),
                                            params={"kernel": ["linear"],
                                                    "C":      np.logspace(-9, 9, num=10000, base=10)},
                                            **generic_args),
              SVC(C=1.0, gamma=0.7, kernel='rbf'), # naiive approach
              SVC(C=1.0, kernel='linear') # naiive approach
              )
    models = list(clf.fit(x_train, y_train.ravel()) for clf in models)

    # title for the plots
    titles = ('RBF kernel','linear kernel', 'Naiive RBF', 'Naiive Linear')# 'Poly Kernel')

    # Set-up 2x2 grid for plotting.
    fig, sub = plt.subplots(2,2)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.ion()

    X0, X1 = iris.data[:, 0], iris.data[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    for clf, title, ax in zip(models, titles, sub.flatten()):
        plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.4)
        ax.scatter(x_train[:, 0], x_train[:, 1], c=y_train.ravel(), cmap=plt.cm.coolwarm, s=20, edgecolors='grey', alpha=0.7)
        ax.scatter(x_test[:, 0], x_test[:, 1], c=y_test.ravel(), cmap=plt.cm.coolwarm, s=20, edgecolors='black', alpha=1)
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xlabel('Sepal length')
        ax.set_ylabel('Sepal width')
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)

    plt.show()