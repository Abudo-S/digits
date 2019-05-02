import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import seaborn as sns



digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=0)

tree = DecisionTreeClassifier(max_depth=10, min_samples_leaf= 1)
tree.fit(X_train, y_train)

print('Accuracy on the training subset: {:.3f}'.format(tree.score(X_train, y_train)))
print('Accuracy on the test subset: {:.3f}'.format(tree.score(X_test, y_test)))




#cv_results = cross_val_score(tree,X_train,y_train,cv=5)
#print('Cross validation result: ' , cv_results)
#print('The mean of Cross validation: ' , np.mean(cv_results))



param_grid = [{'max_depth':np.arange(1, 21),
              'min_samples_leaf':[1, 5, 10, 20, 50, 100]}]

gs = GridSearchCV(estimator=tree, param_grid=param_grid,  cv=10)
gs = gs.fit(X_train, y_train)

#NO NEED FOR get best estimator , because i changed the parameteres in DecisionTreeClassifier decleration
#tree = gs.best_estimator_
print('GridSearchCV BEST params',gs.best_params_)


predictions = tree.predict(X_test)

score = tree.score(X_test, y_test)
print('Accuracy Score: ',score)

cm = metrics.confusion_matrix(y_test, predictions)
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15)
plt.show()