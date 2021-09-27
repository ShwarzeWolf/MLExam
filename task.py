from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.decomposition import PCA

(X_train, y_train), (X_pred, y_pred) = mnist.load_data()

dim = 784
X_train = X_train.reshape(len(X_train), dim)
X_test = X_pred.reshape(len(X_pred), dim)

X_train_first_samples = X_train[:3000, :]
y_train_first_samples = y_train[:3000]

X_train, X_test, y_train, y_test = train_test_split(X_train_first_samples, y_train_first_samples, random_state=30, test_size=0.3)

randomForest = RandomForestClassifier(criterion='gini', min_samples_leaf=10, max_depth=20, n_estimators=10, random_state=30)
randomForestClassifier = OneVsRestClassifier(randomForest).fit(X_train, y_train)
predictions = randomForestClassifier.predict(X_test)
confusionMatrix = confusion_matrix(y_test, predictions)

sum = 0
for i in range(10):
    sum += confusionMatrix[i][i]

#task 3
print(sum)

testData = pd.read_csv('./DataForPrediction_FinalTask.csv')
testData = testData.iloc[:, 1:]

predictedValues = randomForestClassifier.predict_proba(testData)
#task5
print(max(predictedValues[73]))

pca = PCA(n_components=40, svd_solver='full')
modelPCA = pca.fit(X_train)
X_train = modelPCA.transform(X_train)
X_test = modelPCA.transform(X_test)


decisionTree = DecisionTreeClassifier(criterion='gini', min_samples_leaf=10, max_depth=20, random_state=30)
decisionTreeClassifier = OneVsRestClassifier(decisionTree).fit(X_train, y_train)
predictions = decisionTreeClassifier.predict(X_test)
confusionMatrix = confusion_matrix(y_test, predictions)

sum = 0
for i in range(10):
    sum += confusionMatrix[i][i]

#task 8
print(sum)

testData = modelPCA.transform(testData)

predictedValues = decisionTreeClassifier.predict_proba(testData)
#task9
print(predictedValues[73])