import os
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_classification_model(folder_path):
    true_file_path = os.path.join(folder_path, 'labels.npy')
    
    pred_file_path = os.path.join(folder_path, 'pred_labels.npy')

    true_data = np.load(true_file_path)
    pred_data = np.load(pred_file_path)

    accuracy = accuracy_score(true_data, pred_data)
    precision = precision_score(true_data, pred_data, average='binary')
    recall = recall_score(true_data, pred_data, average='binary')
    f1 = f1_score(true_data, pred_data, average='binary')

    # report = classification_report(true_data, pred_data, output_dict=True, zero_division=0)

    # accuracy = report['accuracy']
    # precision = report['1']['precision']
    # recall = report['1']['recall']
    # f1_score = report['1']['f1-score']

    return accuracy, precision, recall, f1

dirs = [
    "results/test/mdhp-net-0.4-4-N128/statistics",
    "results/test/sissa-lstm-0.4-4-N128/statistics",
    "results/test/sissa-rnn-0.4-4-N128/statistics",
    "results/test/sissa-cnn-0.4-4-N128/statistics"
]

for dir in dirs:
    model_name = os.path.basename(os.path.dirname(dir))
    accuracy_result, precision_result, recall_result, f1_score_result = evaluate_classification_model(dir)
    
    print(f"Metrics for {model_name}:")
    print(f"  Accuracy: {accuracy_result}")
    print(f"  Precision: {precision_result}")
    print(f"  Recall: {recall_result}")
    print(f"  F1 Score: {f1_score_result}")
