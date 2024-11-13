import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from matplotlib import font_manager
from matplotlib import pyplot as plt

# Load Times New Roman font in case it is not installed
font_dirs = ["./fonts/Times New Roman"]
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
for font_file in font_files:
    font_manager.fontManager.addfont(font_file)
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 16
plt.rcParams["font.style"] = "normal"

models = ["MDHP-Net", "SISSA-LSTM", "SISSA-RNN", "SISSA-CNN"]

folder_paths = [
    "results/test/MDHP_NET_N128/statistics",
    "results/test/SISSA_LSTM_N128/statistics",
    "results/test/SISSA_RNN_N128/statistics",
    "results/test/SISSA_CNN_N128/statistics",
]

metrics_values = {model: [] for model in models}

for folder_path, model in zip(folder_paths, models):
    true_labels = np.load(os.path.join(folder_path, "labels.npy"))
    pred_labels = np.load(os.path.join(folder_path, "pred_labels.npy"))

    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, average="binary")
    recall = recall_score(true_labels, pred_labels, average="binary")
    f1 = f1_score(true_labels, pred_labels, average="binary")

    metrics_values[model] = [accuracy, precision, recall, f1]

metrics = ["Accuracy", "Precision", "Recall", "F1-score"]

colors = ["#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F"]
bar_width = 0.18
index = np.arange(len(models))

plt.figure(figsize=(10, 6))

for i, metric in enumerate(metrics):
    values = [metrics_values[model][i] for model in models]
    plt.bar(
        index + i * bar_width,
        values,
        bar_width - 0.05,
        label=metric,
        color=colors[i],
    )


for i, metric in enumerate(metrics):
    values = [metrics_values[model][i] for model in models]
    for j, value in enumerate(values):
        y_offset = 0 if i in [0, 2] or j != 1 else 3e-3
        y_offset = y_offset if i in [0, 2] or j !=0 else 2e-3
        plt.text(
            index[j] + i * bar_width,
            value + y_offset,
            "{:.3f}".format(value),
            ha="center",
            va="bottom",
            fontsize=14,
        )

plt.xticks(index + bar_width * (len(metrics) - 1) / 2, models)
plt.ylim(0.9, 1.0)

plt.legend(loc="upper right", borderaxespad=0.5, fontsize=16)
plt.tight_layout()

plt.savefig("results/visualization/bar_chart_model_comparison.png")
plt.savefig("results/visualization/bar_chart_model_comparison.pdf")
