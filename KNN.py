from sklearn.datasets import load_digits
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score 


digits = load_digits()


x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=0)


knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train,y_train)

param_grid = {'n_neighbors': np.arange(1, 25)}
knn_GS = GridSearchCV(knn, param_grid, cv=5)
knn_GS.fit(x_train,y_train)
print(knn_GS.best_params_)

#cv_results = cross_val_score(knn, x_train, y_train, cv=5) 
#print('Cross validation result: ' , cv_results)
#print('The mean of Cross validation: ' , np.mean(cv_results))

predictions = knn_GS.predict(x_test)

score = knn.score(x_test, y_test)
print('Accuracy Score: ',score)

cm = metrics.confusion_matrix(y_test, predictions)
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15);