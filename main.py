'''

	Assumptions
	-----------

	1. Mid sem copies collected in 2nd or 3rd attempt can be
	if student forgot collecting the copy or if the student
	had given for recheck

	2. We remove the two withdrawn cases (id 204, 205)

	3. quiz1 & quiz2 for 30 marks each, part a and part b for 40 marks each


'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
# from sklearn.

attributes = ['idno', 'year', 'attendance', 'gender', 'cgpa', 'midsem', 'midsemgrade', 'midsemcollection', 'quiz1', 'quiz2', 'parta', 'partb', 'grade']
# garbage = ',,,,,,,,,,,,,,\n'
grades = ['NC', 'I', '','E', 'D', 'C-', 'C', 'B-', 'B', 'A-', 'A']

def get(line, attr):
	global attributes
	i = attributes.index(attr)
	return line[i]

# LOAD CLEAN DATA
with open('data_new.csv', 'r') as f:
	lines = f.readlines()

# remove withdrawn cases
lines = lines[:-2]

# remove the garbage character at the title
lines[0] = lines[0][3:]

# remove '\n' and construct list of values
for i in range(len(lines)):
	lines[i] = lines[i][:-1].split(',')

# for line in lines:
	# print(line)

def getNonZero(lines, cols):

	vals = []

	for l in lines:
		val = []
		allright = True
		for i in cols:
			if(get(l, i)==''):
				allright = False
				break
			else:
				val.append(get(l, i))
		if(allright):
			vals.append(val)

	return vals

'''
	Experiment 1: Correlation matrices of various scores
'''

# for key in attributes:
# 	content = getNonZero(lines, [key])
# 	print(key, ':', len(content))
#
# content = getNonZero(lines, ['midsem', 'midsemgrade', 'midsemcollection', 'quiz1', 'quiz2', 'parta', 'partb', 'grade'])
#
# data = []
# for i in range(1, len(content)):
# 	content[i][1] = grades.index(content[i][1])
# 	content[i][-1] = grades.index(content[i][-1])
# 	data.append(content[i])
#
# data = np.array(content[1:], dtype=np.float32)
# print('\nData\n', data)
#
# for i in range(data.shape[1]):
# 	print(data[:, i].mean(), data[:, i].std())
# 	data[:, i] = (data[:, i] - data[:, i].mean()) / (data[:, i].std())
#
# print('\nNormalized Data\n', data)
#
# cov = np.cov(data.T)
#
# print('\ncorrelation\n', np.array(cov*100, dtype=np.int32)/100.0)

'''
	Experiment 2: Mutual Information among the variables
'''

def getBucketedNonZeroData(lines, cols):

	content = getNonZero(lines, cols)

	# print(content)

	for i, c in enumerate(cols):
		if c in ['midsem', 'quiz1', 'quiz2', 'parta', 'partb']:
			for idx in range(1, len(content)):
				# print(content[idx][i])
				content[idx][i] = int(float(content[idx][i])/5)

	return content

#TODO
# content = getBucketedNonZeroData(lines, ['midsem', 'midsemgrade', 'quiz1', 'quiz2', 'parta', 'partb', 'grade'])

'''
	Experiment 3: Classification Comparison
'''

test = 1

# test 1
if test==1:
	content = getNonZero(lines, ['midsem', 'quiz1', 'quiz2', 'parta', 'partb', 'grade'])

# test 2
elif test==2:
	content = getNonZero(lines, ['year', 'attendance', 'cgpa', 'grade'])

# test 3
elif test==3:
	content = getNonZero(lines, ['midsem', 'quiz1', 'quiz2', 'parta', 'partb', 'year', 'attendance', 'cgpa', 'grade'])

# a = []
# for l in content[1:]:
# 	a.append(l[2])
# a = list(map(float, a))
# # print(a)
# # print('mean:', np.mean(a))
# # print('std:', np.std(a))


# ----------------------------------------------------------------------------------
# train_x = []
# train_y = []
#
# for c in content[1:]:
# 	train_x.append(c[:-1])
# for c in content[1:]:
# 	train_y.append(c[-1])
#
# for i in range(len(train_x)):
# 	# print(train_y)
# 	# train_x[i][1] = grades.index(train_x[i][1])
#
# 	if test==1:
# 		train_y[i] = grades.index(train_y[i])
# 		train_x[i][0] = float(train_x[i][0])
# 		for k in range(1,len(content[0])-1):
# 			# print(k)
# 			train_x[i][k] = float(train_x[i][k])
# 	elif test==2:
# 		train_y[i] = grades.index(train_y[i])
# 		for k in range(len(content[0])-1):
# 			train_x[i][k] = float(train_x[i][k])
# 	elif test==3:
# 		train_y[i] = grades.index(train_y[i])
# 		for k in range(len(content[0])-1):
# 			train_x[i][k] = float(train_x[i][k])
#
# # print(len(train_x))
# # Decision Tree
# cfiers = [DecisionTreeClassifier(), GaussianNB(), SVC(kernel='rbf'), SVC(kernel='linear'), SVC(kernel='sigmoid')]
# cfiernames = ['Decision tree', 'Naive Bayes', 'RBF SVM', 'Linear SVM', 'Sigmoid SVM']
#
# for i in range(1,21):
# 	cfiers.append(KNeighborsClassifier(n_neighbors=i))
# 	cfiernames.append('KNN ' + str(i))
#
# cfierscores = []
# skf = StratifiedKFold(n_splits=10)
# skf.get_n_splits(train_x, train_y)
#
# # print(train_x, train_y)
# train_x = np.array(train_x)
# train_y = np.array(train_y)
#
# for i, cfier in enumerate(cfiers):
# 	scores = []
#
# 	for train_index, test_index in skf.split(train_x, train_y):
# 		# print("TRAIN:", train_index, "TEST:", test_index)
# 		X_train, X_test = train_x[train_index], train_x[test_index]
# 		y_train, y_test = train_y[train_index], train_y[test_index]
#
# 		# print(X_train)
#
# 		cfier.fit(X_train, y_train)
# 		s = cfier.score(X_test, y_test)
# 		scores.append(s)
#
# 	print('\n', cfiernames[i])
# 	print('mean:', np.mean(scores))
# 	print('std:', np.std(scores))
# 	cfierscores.append( (np.mean(scores), np.std(scores)) )
# # print(cfierscores)
#
#
# # 5 cfiers
# means = []
# stds = []
#
# for s in cfierscores[0:5]:
# 	means.append(s[0])
# 	stds.append(s[1])
#
# plt.plot(means)
# plt.errorbar(x=range(5), y=means, yerr=stds, linestyle='None', marker='^')
# plt.xticks(range(5), cfiernames)
# plt.ylabel('Mean accuracy of 5-fold CV')
# plt.xlabel('Classifier')
# plt.show()
#
# # knn
# means = []
# stds = []
#
# for s in cfierscores[5:]:
# 	means.append(s[0])
# 	stds.append(s[1])
#
# plt.plot(means)
# plt.errorbar(x=range(20), y=means, yerr=stds, linestyle='None', marker='^')
# plt.xticks( range(20), range(1, 21) )
# plt.ylabel('Mean accuracy of 5-fold CV')
# plt.xlabel('Value of k for K-NN')
# plt.show()

# ----------------------------------------------------------------------------------

# PCA
train_x = []
train_y = []

for c in content[1:]:
	train_x.append(c[:-1])
for c in content[1:]:
	train_y.append(c[-1])

for i in range(len(train_x)):
	# print(train_y)
	# train_x[i][1] = grades.index(train_x[i][1])

	if test==1:
		train_y[i] = grades.index(train_y[i])
		train_x[i][0] = float(train_x[i][0])
		for k in range(1,len(content[0])-1):
			# print(k)
			train_x[i][k] = float(train_x[i][k])
	elif test==2:
		train_y[i] = grades.index(train_y[i])
		for k in range(len(content[0])-1):
			train_x[i][k] = float(train_x[i][k])
	elif test==3:
		train_y[i] = grades.index(train_y[i])
		for k in range(len(content[0])-1):
			train_x[i][k] = float(train_x[i][k])

# print(train_x)
# print(train_y)

# print(len(train_x))
# Decision Tree
cfiers = [DecisionTreeClassifier(), GaussianNB(), SVC(kernel='rbf'), SVC(kernel='linear'), SVC(kernel='sigmoid')]
cfiernames = ['Decision tree', 'Naive Bayes', 'RBF SVM', 'Linear SVM', 'Sigmoid SVM']

for i in range(1,21):
	cfiers.append(KNeighborsClassifier(n_neighbors=i))
	cfiernames.append('KNN ' + str(i))

cfierscores = []
skf = StratifiedKFold(n_splits=10)
skf.get_n_splits(train_x, train_y)

# print(train_x, train_y)
train_x = np.array(train_x)
train_y = np.array(train_y)

pca = PCA(n_components=1)

for i, cfier in enumerate(cfiers):
	scores = []

	for train_index, test_index in skf.split(train_x, train_y):
		# print("TRAIN:", train_index, "TEST:", test_index)
		X_train, X_test = train_x[train_index], train_x[test_index]
		y_train, y_test = train_y[train_index], train_y[test_index]

		X_train = pca.fit_transform(X_train)
		X_test = pca.fit_transform(X_test)

		# print(X_train)

		cfier.fit(X_train, y_train)
		s = cfier.score(X_test, y_test)
		scores.append(s)

	print('\n', cfiernames[i])
	print('mean:', np.mean(scores))
	print('std:', np.std(scores))
	cfierscores.append( (np.mean(scores), np.std(scores)) )
# print(cfierscores)


# 5 cfiers
means = []
stds = []

for s in cfierscores[0:5]:
	means.append(s[0])
	stds.append(s[1])

plt.plot(means)
plt.errorbar(x=range(5), y=means, yerr=stds, linestyle='None', marker='^')
plt.xticks(range(5), cfiernames)
plt.ylabel('Mean accuracy of 5-fold CV')
plt.xlabel('Classifier')
plt.show()

# knn
means = []
stds = []

for s in cfierscores[5:]:
	means.append(s[0])
	stds.append(s[1])

plt.plot(means)
plt.errorbar(x=range(20), y=means, yerr=stds, linestyle='None', marker='^')
plt.xticks( range(20), range(1, 21) )
plt.ylabel('Mean accuracy of 5-fold CV')
plt.xlabel('Value of k for K-NN')
plt.show()




'''
content = getNonZero(lines, ['midsem', 'midsemgrade', 'quiz1', 'quiz2', 'parta', 'partb', 'grade'])

data = []
for i in range(1, len(content)):
	content[i][1] = grades.index(content[i][1])
	content[i][6] = grades.index(content[i][6])
	data.append(content[i])

data = np.array(data, dtype=np.float32)
print('\nData\n', data)

for i in range(data.shape[1]):
	print(data[:, i].mean(), data[:, i].std())
	data[:, i] = (data[:, i] - data[:, i].mean()) / (data[:, i].std())

print('\nNormalized Data\n', data)

cov = np.cov(data.T)

print('\ncovariance\n', np.array(cov*100, dtype=np.int32)/100.0)
'''
