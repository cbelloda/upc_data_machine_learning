from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier


def knn(features,class_feature,neighbors=3,testdata=None):
  neigh = KNeighborsClassifier(neighbors)
  neigh.fit(features, class_feature)  
  if testdata is None:
    testdata=features  
  predict = neigh.predict(testdata)
  conf_mat_knn = confusion_matrix(class_feature, neigh.predict(features))

  return {"predict":predict,"mat_conf":conf_mat_knn,"model":neigh}


def decision_tree(features,class_feature,testdata=None,max_depth=3,criterion="entropy"):
  clf = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth)
  clf.fit(features, class_feature)
  if testdata is None:
    testdata=features  
  predict = clf.predict(testdata)
  conf_mat_knn = confusion_matrix(class_feature, clf.predict(features))

  return {"predict":predict,"mat_conf":conf_mat_knn,"model":clf}