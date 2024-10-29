import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

csv = 'Compiled Annotations CSV.csv'
data = pd.read_csv(csv, encoding='ISO-8859-1')

X = data.drop(columns='Label')
y = data['Label']

X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.3, random_state=42)


dummy_model = DummyClassifier(strategy='most_frequent', random_state=42)
dummy_model.fit(X_train, y_train)

# #Make Predictions
Model_1_Predictions = dummy_model.predict(X_dev)

# Model_1_Predictions = Model_1.predict(X_test)


# # #Display Results
print(f"dummy_model: {accuracy_score(y_dev, Model_1_Predictions)}")

# print(f"Model 1: {accuracy_score(y_test, dummy_model_Predictions)}")

print("dummy model")
print(classification_report(y_dev, Model_1_Predictions))

# print("")
# print(classification_report(y_test, dummy_model_Predictions))