import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression


class Model:
    def __init__(self):
        # Initialize pre-trained BERT
        self.model_name = 'bert-base-uncased'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.eval()
        self.clf = LogisticRegression(max_iter=1000)

    def fit(self, X, y):
        # Tokenize and encode the text inputs
        inputs = self.tokenizer(X, padding=True, truncation=True, max_length=128, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].numpy()

        # Train the logistic regression model on the transformed data
        self.clf.fit(embeddings, y)

    def predict(self, X):
        # Transform the new text data using the trained BERT model
        inputs = self.tokenizer(X, padding=True, truncation=True, max_length=128, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].numpy()

        # Predict labels
        return self.clf.predict(embeddings)