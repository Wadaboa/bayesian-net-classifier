# Bayesian network classifiers

In this work, we tested the capabilities of various Bayesian networks structures (mainly Naive Bayes and augmented Naive Bayes) in a classification task, over the standard Adult dataset, which aims at separating people whose income is greater than 50 thousands dollars per year from the rest.

## Installation & Execution

In order to play with the provided Jupyter notebook and test the various classifiers, it is necessary to follow these steps:
* Install `Python 3.8` on your system
* Optionally create a virtual environment in the root directory of the project (`python3 -m venv venv`) and activate it (`source venv/bin/activate`)
* Install the required dependencies (`pip install -r requirements.txt`)

## Implemented models

* **Naive Bayes** (NB): Implementation given by [`pgmpy`](https://github.com/pgmpy/pgmpy)
* **Tree-Augmented Naive Bayes** (TAN): Implementation taken by a pending pull request on the [`pgmpy`](https://github.com/pgmpy/pgmpy/pull/1266) repository
* **BN-Augmented Naive Bayes** (BAN): Custom implementation (slow and buggy)
* **Forest-Augmented Naive Bayes** (FAN): Custom implementation

## Source files structure

The Adult dataset was downloaded from the [`UCI Machine Learning Repository`](http://archive.ics.uci.edu/ml/datasets/adult) and placed inside the `dataset` folder.

The whole project was written in the Jupyter notebook [`classify.ipynb`](classify.ipynb), while the custom structural learning algorithms are implemented in the [`estimators.py`](estimators.py) file. 

Moreover, a complete overview of the whole data pre-processing, classification and evaluation pipeline can be found in the [`report.pdf`](report/report.pdf) file, inside the `report` folder.
