# Adaboost code was used from https://github.com/srijarkoroy/adaboost

import matplotlib.pyplot as plt
from datagenerate import dataset
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

X, y, label_encoder = dataset()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=200,
    random_state=42,
    algorithm='SAMME'
)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {accuracy:.2%}")

errors = []
for y_pred_iter in clf.staged_predict(X_train):
    error = 1 - accuracy_score(y_train, y_pred_iter)
    errors.append(error)

plt.plot(range(1, len(errors) + 1), errors)
plt.xlabel('Number of Estimators')
plt.ylabel('Training Error')
plt.title('Training Error vs. Number of Estimators')
plt.show()