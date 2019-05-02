from sklearn.datasets import load_digits
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import GridSearchCV

digits = load_digits()

x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=0)

# all parameters not specified are set to their defaults
logisticRegr = LogisticRegression()
logisticRegr.fit(x_train, y_train)

#cv_results = cross_val_score(logisticRegr, x_train, y_train, cv=5) 
#print('Cross validation result: ' , cv_results)
#print('The mean of Cross validation: ' , np.mean(cv_results))

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10], "penalty":["l1","l2"]}
grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid.fit(x_train, y_train)

print("Best cross-validation score: {:.2f}".format(grid.best_score_))
print("Best parameters: ", grid.best_params_)
print("Best estimator: ", grid.best_estimator_)
# Make predictions on entire test data
#predictions = logisticRegr.predict(x_test)
predictions = grid.predict(x_test)

#lr = grid.best_estimator_
#lr.fit(x_train, y_train)
#lr.predict(x_test)
score=grid.score(x_test, y_test)
print("Score: {:.2f}".format(score))

# Use score method to get accuracy of model
#score = logisticRegr.score(x_test, y_test)
#print(score)

#Confusion Matrix using seaborn

cm = metrics.confusion_matrix(y_test, predictions)

plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15);




# Print to show there are 1797 images (8 by 8 images for a dimensionality of 64)
#print("Image Data Shape" , digits.data.shape)

# Print to show there are 1797 labels (integers from 0-9)
#print("Label Data Shape", digits.target.shape)

#plt.figure(figsize=(20,4))
#for index, (image, label) in enumerate(zip(digits.data[0:5], digits.target[0:5])):
    #plt.subplot(1, 5, index + 1)
    #plt.imshow(np.reshape(image, (8,8)), cmap=plt.cm.gray)
    #plt.title('Training: %i\n' % label, fontsize = 20)