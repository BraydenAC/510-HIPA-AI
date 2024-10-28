import csv
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the CSV
csv_file = 'Compiled Annotations Distribution.csv'
data = pd.read_csv(csv_file, encoding='ISO-8859-1')

#Split data into X and y
X = data.drop(columns='Label')
y = data['Label']

# Split the data into train, dev, and test
X_train, X_dev_test, y_train, y_dev_test = train_test_split(X, y, test_size=0.2)
X_dev, X_test, y_dev, y_test = train_test_split(X_dev_test, y_dev_test, test_size=0.5)

# Convert to DataFrames for easier CSV export
train_df = pd.concat([X_train, y_train], axis=1)
dev_df = pd.concat([X_dev, y_dev], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

# Write to CSV files
train_df.to_csv("train.csv", index=False)
dev_df.to_csv("dev.csv", index=False)
test_df.to_csv("test.csv", index=False)