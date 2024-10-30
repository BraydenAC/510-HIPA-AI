import json
import os
import sys
import time

import numpy as np
from model import Model

input_dir = '/app/input_data/'
output_dir = '/app/output/'
program_dir = '/app/program'
submission_dir = '/app/ingested_program'

sys.path.append(program_dir)
sys.path.append(submission_dir)


def get_training_data():
    dataSet = np.genfromtxt(os.path.join(input_dir, 'train.csv'), delimiter=',', skip_header=1)
    X_train = dataSet[:, :-1]
    y_train = dataSet[:, -1]
    return X_train, y_train


def get_prediction_data():
    return np.genfromtxt(os.path.join(input_dir, 'test_data.csv'), delimiter=',', skip_header=1)


def main():
    print('Reading Data')
    X_train, y_train = get_training_data()
    X_test = get_prediction_data()
    print('-' * 10)
    print('Starting')
    start = time.time()
    m = Model()
    print('-' * 10)
    print('Training Model')
    m.fit(X_train, y_train)
    print('-' * 10)
    print('Running Prediction')
    prediction = m.predict(X_test)
    duration = time.time() - start
    print('-' * 10)
    print(f'Completed Prediction. Total duration: {duration}')
    np.savetxt(os.path.join(output_dir, 'prediction'), prediction, delimiter=',')
    with open(os.path.join(output_dir, 'metadata.json'), 'w+') as f:
        json.dump({'duration': duration}, f)
    print()
    print('Ingestion Program finished. Moving on to scoring')


if __name__ == '__main__':
    main()
