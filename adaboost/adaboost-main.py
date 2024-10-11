# Adaboost code was used from https://github.com/srijarkoroy/adaboost

import matplotlib.pyplot as plt
import numpy as np
from datagenerate import dataset
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def print_model_predictions(y_pred_labels, y_test_labels):
    max_idx_width = max(len(str(len(X_test))), 5)
    max_features_width = 30  
    max_pred_label_width = 20  
    max_true_label_width = 20 

    with open("output.txt", "w") as file:
        header = f"{'Input':<{max_idx_width}} | {'IR Sensor Values':<{max_features_width}} | {'Predicted':<{max_pred_label_width}} | {'Actual':<{max_true_label_width}}\n"
        file.write(header)
        file.write('-' * len(header) + "\n")

        # Loop and file.write formatted output
        for idx, (features, pred_label, true_label) in enumerate(zip(X_test, y_pred_labels, y_test_labels)):

            features_str = np.array2string(features, separator=',', max_line_width=max_features_width).replace('\n', '')
            # Format the strings with padding
            file.write(f"{idx:<{max_idx_width}} | {features_str:<{max_features_width}} | {pred_label:<{max_pred_label_width}} | {true_label:<{max_true_label_width}}\n")
        
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

y_pred_labels = label_encoder.inverse_transform(y_pred)
y_test_labels = label_encoder.inverse_transform(y_test)

print_model_predictions(y_pred_labels,y_test_labels)

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