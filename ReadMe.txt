
The assignment was done only by me Angely =)

*****************
Second Part Notes
*****************

I started to balance the samples for each classes, this gave me a notorious difference with respect to unbalanced classes,
I chose the number of samples similar to the shortest class

The estimator chosen was LinearSVC, based on a GridSearch. 
The parameter  "weighted classes" = balanced increased the F1 macro in 1% 
The Linear Regression gave me worst results than LinearSVC
I also tried with SVR but it always gave me an error of Convergence

I let here the code of the LR and SVR estimators used before in order to choose the best ( LinearSVC code is in the notebook)

#############################################
# Clasifier LOGISTTIC REGRESION
#############################################

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

#Hyper tuning parameters
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}
grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5)
grid.fit(X_train, Y_train)
print("Best cross-validation score: {:.2f}".format(grid.best_score_))
print("Best parameters: ", grid.best_params_)
print("Best estimator: ", grid.best_estimator_)

###############
# SVR
###############
from sklearn.svm import SVR

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}
grid = GridSearchCV(SVR(max_iter=1000), param_grid, cv=5)
grid.fit(X_train, Y_train)
print("Best cross-validation score: {:.2f}".format(grid.best_score_))
print("Best parameters: ", grid.best_params_)
print("Best estimator: ", grid.best_estimator_)


Lemmatizaction vs stemming? 
I tried with both, and lemmatization gave me better results

PreProcesing?
Deleting non useful characters such as: punctuation, blank spaces, stop words and numbers
did not help in anything, indeed they decreased the accuracy, for that reason I did not use them.

Note: Deleting html characters did not change the previous results, then better to not use it 
>>> Only for practicing I pre processed  the data  deleting: stop words, blank spaces, symbols and html code 
but the F1 accuracy and macro decreased in 1 % 

Vectorizer?
I worked with TfId vectorizer and important parameters found were:
ngram_range=(1,2) and min_df=5
The use of both gave me the best results, I read about these values in some papers and blogs

Some errors in the code and how they were solved 

1) When I ran the Logistic Regression ( 1st part), it showed an error about the target's data type
Error: ufunc 'rint' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''
Solution: Adding the data type to both Y train and test, like is shown below
Y_train = np.array(df_train['score'].tolist(), dtype=np.int64)
Y_test = np.array(df_test['score'].tolist(), dtype=np.int64)
Source: https://stackoverflow.com/a/48919314

2) NLTK STEMM
open conda console and run the below:
import nltk
nltk.download()
Source: https://www.nltk.org/data.html


Some Papers and links useful
http://cs229.stanford.edu/proj2014/Chen%20Li,%20Jin%20Zhang,%20Prediction%20of%20Yelp%20Review%20Star%20Rating%20using%20Sentiment%20Analysis.pdf
Kaggle source
https://www.kaggle.com/docktorrr/logistic-regression-tf-idf-n-grams-and-stemming
