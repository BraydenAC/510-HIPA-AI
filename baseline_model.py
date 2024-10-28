import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyClassifier

# Load the CSV
csv_file = 'Compiled Annotations Distribution.csv'
data = pd.read_csv(csv_file, encoding='ISO-8859-1')

label_column = 'Label'

# Calculate the distribution of labels
label_counts = data[label_column].value_counts()

# Print the distribution
print("Label distribution:")
print(label_counts)
print(label_counts.shape)

dummy_model = DummyClassifier(strategy='most_frequent', random_state=42)
dummy_model.fit(X_train, y_train)