from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
cancer = load_breast_cancer()
X_train,X_test,y_train,y_test = train_test_split(cancer.data,cancer.target,stratify=cancer.target,random_state=23)

DTC =DecisionTreeClassifier().fit(X_train,y_train)

accuracy = cross_val_score(DTC, X_test, y_test, scoring='accuracy', cv=5)
print("准确率:",accuracy.mean())


