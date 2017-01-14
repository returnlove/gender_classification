from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np


def main():
	
	# Data and labels
	X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

	y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']


	# Initializing the classifiers

	clf_lr = LogisticRegression()
	clf_dt = DecisionTreeClassifier()
	clf_rf = RandomForestClassifier()
	clf_svm = SVC()
	clf_knn = KNeighborsClassifier()



	# fitting the models

	clf_lr.fit(X,y)
	clf_dt.fit(X,y)
	clf_rf.fit(X,y)
	clf_svm.fit(X,y)
	clf_knn.fit(X,y)

	# predicting on the same dataset

	def check_accuracy(model):
		clf = model
		return accuracy_score(clf.predict(X), y) * 100

	acc_lr = check_accuracy(clf_lr)
	acc_dt = check_accuracy(clf_dt)
	acc_rf = check_accuracy(clf_rf)
	acc_svm = check_accuracy(clf_svm)
	acc_knn = check_accuracy(clf_knn)

	print(acc_lr)
	print(acc_dt)
	print(acc_rf)
	print(acc_svm)
	print(acc_knn)

	index = np.argmax([acc_knn, acc_lr, acc_dt, acc_rf, acc_svm])
	classifiers = {0: 'knn', 1: 'lr', 2: 'dt', 3: 'rf', 4: 'svm'}
	# print(index)
	print('The best classifier is: ',classifiers[index])

if __name__ == "__main__":
	main()