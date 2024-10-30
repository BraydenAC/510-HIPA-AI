# from sklearn.cluster import KMeans
#
#
# class Model:
#     def __init__(self):
#         self.kmeans = KMeans(n_clusters=3)
#
#     def fit(self, X, y):
#         self.kmeans.fit(X=X, y=y)
#
#     def predict(self, X):
#         return self.kmeans.predict(X)



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV
csv_file = '/content/Compiled Annotations Distribution.csv'
data = pd.read_csv(csv_file, encoding='ISO-8859-1')

label_column = 'Label'

# Calculate the distribution of labels
label_counts = data[label_column].value_counts()

# Print the distribution
print("Label distribution:")
print(label_counts)

# Plot the distribution
plt.figure(figsize=(10, 6))
sns.countplot(data=data, x=label_column, order=label_counts.index)
plt.title("Label Distribution")
plt.xlabel("Labels")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.show()


# Import libraries
import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score

# Load the csv with proper encoding
df = pd.read_csv('/content/Compiled Annotations CSV.csv', encoding='ISO-8859-1')

# Extract texts and labels
texts = df['Features'].tolist()
labels = df['Label'].tolist()

# Load BERT
model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Set the model to evaluation mode
model.eval()

# Tokenize and encode the text inputs
inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors='pt')

# Pass the inputs through the BERT model
with torch.no_grad():
    outputs = model(**inputs)

# Extract the sentence embeddings
sentence_embeddings = outputs.last_hidden_state[:, 0, :].numpy()  # shape: [batch_size, hidden_state_size]

# Save embeddings to avoid recomputation
pd.DataFrame(sentence_embeddings).to_csv('sentence_embeddings.csv', index=False)
pd.DataFrame(labels, columns=['Label']).to_csv('labels.csv', index=False)

# Load embeddings and labels
embeddings = pd.read_csv('sentence_embeddings.csv').values
labels = pd.read_csv('labels.csv').values.ravel()

# Split the data
X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.3, random_state=42)

# Train the logistic regression model
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Calculate precision, recall, f1 score
precision = precision_score(y_test, y_pred, pos_label='Yes')
recall = recall_score(y_test, y_pred, pos_label='Yes')
f1 = f1_score(y_test, y_pred, pos_label='Yes')

print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')

# Detailed classification report
print(classification_report(y_test, y_pred, target_names=['No', 'Yes']))

#%% md
Cross validation analysis:
#%%
from sklearn.model_selection import cross_val_score

scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
print(f'Cross-validated accuracy: {scores.mean():.2f}')
