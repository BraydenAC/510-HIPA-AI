import json
import os
import numpy as np
from sklearn.metrics import f1_score

reference_dir = os.path.join('/app/input/', 'ref')
prediction_dir = os.path.join('/app/input/', 'res')
score_dir = '/app/output/'

print('Reading prediction')
prediction = np.genfromtxt(os.path.join(prediction_dir, 'prediction.csv'), delimiter=',', skip_header=1)
truth = np.genfromtxt(os.path.join(reference_dir, 'test_label.csv'), delimiter=',', skip_header=1)
with open(os.path.join(prediction_dir, 'metadata.json')) as f:
    duration = json.load(f).get('duration', -1)

prediction = prediction.astype(int)
truth = truth.astype(int)

print('Checking Accuracy')
f1 = f1_score(truth, prediction, pos_label=1)
print('Scores:')
scores = {
    'f1_score': f1,
    'duration': duration
}
print(scores)

with open(os.path.join(score_dir, 'scores.json'), 'w') as score_file:
    score_file.write(json.dumps(scores))
