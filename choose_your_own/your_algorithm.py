#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
#plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary

from time import time

#from sklearn import tree
#clf = tree.DecisionTreeClassifier(min_samples_split=40)

#from sklearn.ensemble import AdaBoostClassifier
# 1a tentativa - Accuracy 0.924
#clf = AdaBoostClassifier(n_estimators=100)

# 2a tentativa - Accuracy 0.916
#clf = AdaBoostClassifier(n_estimators=200)

# 3a tentativa - Accuracy 0.924
#from sklearn.tree import DecisionTreeClassifier
#clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=200, algorithm="SAMME")

# 4a tentativa - Accuracy 0.928
#clf = AdaBoostClassifier(n_estimators=20)


# 5a tentativa - Accuracy 0.916
#clf = AdaBoostClassifier(n_estimators=10)

## 6a tentativa - Accuracy 0.928
#clf = AdaBoostClassifier(n_estimators=15)

# 7a tentativa - Accuracy 0.932
#clf = AdaBoostClassifier(n_estimators=15, learning_rate=0.3)

# 8a tentativa - Accuracy 0.932
#from sklearn.tree import DecisionTreeClassifier
#clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=4), n_estimators=15, learning_rate=0.3)

# 9a tentativa - Accuracy 0.
#from sklearn.tree import DecisionTreeClassifier
#clf = AdaBoostClassifier(SVM(max_depth=4), n_estimators=15, learning_rate=0.3) N terminada!!

#---------------

#from sklearn.neighbors import KNeighborsClassifier
# 1a tentativa - Accuracy 0.936
#clf = KNeighborsClassifier(n_neighbors=3)

# 2a tentativa - Accuracy 0.94
#clf = KNeighborsClassifier(n_neighbors=1)

# 3a tentativa - Accuracy 0.928
#clf = KNeighborsClassifier(n_neighbors=2)

# 4a tentativa - Accuracy 0.94
#clf = KNeighborsClassifier(n_neighbors=4)

# 5a tentativa - Accuracy 0.92
#clf = KNeighborsClassifier(n_neighbors=5)

#---------------

from sklearn.ensemble import RandomForestClassifier
# 1a tentativa - Accuracy 0.92
#clf = RandomForestClassifier(max_depth=2, random_state=0)

# 2a tentativa - Accuracy 0.912
#clf = RandomForestClassifier(random_state=0)

# 3a tentativa - Accuracy 0.916
#clf = RandomForestClassifier(max_depth=6, random_state=0)

# 4a tentativa - Accuracy 0.0932
clf = RandomForestClassifier(max_depth=4, n_estimators=8)


t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

t1 = time()
pred = clf.predict(features_test)
print "testing time:", round(time()-t1, 3), "s"

from sklearn.metrics import accuracy_score
print accuracy_score(labels_test, pred)


try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
