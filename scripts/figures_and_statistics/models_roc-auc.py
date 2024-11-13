import os
import numpy as np
from sklearn.metrics import roc_curve, auc
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from matplotlib import font_manager
from matplotlib import pyplot as plt

# Load Times New Roman font in case it is not installed
font_dirs = ["./fonts/Times New Roman"]
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
for font_file in font_files:
    font_manager.fontManager.addfont(font_file)
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 16

folder_paths = [
    "results/test/MDHP_NET_N128/statistics",
    "results/test/SISSA_LSTM_N128/statistics",
    "results/test/SISSA_RNN_N128/statistics",
    "results/test/SISSA_CNN_N128/statistics",
]

fig, axs = plt.subplots(2, 2, figsize=(11, 8))

models = ["MDHP-Net", "SISSA-LSTM", "SISSA-RNN", "SISSA-CNN"]

for i, folder_path in enumerate(folder_paths):
    true_labels_binary = np.load(os.path.join(folder_path, "labels.npy"))
    pred_probabilities = np.load(os.path.join(folder_path, "preds.npy"))[:, 1]

    fpr, tpr, _ = roc_curve(true_labels_binary, pred_probabilities)

    roc_auc = auc(fpr, tpr)

    ax = axs[i // 2, i % 2]
    ax.plot(fpr, tpr, color="red", lw=2, label="ROC curve (area = %0.2f)" % roc_auc)
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.1])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")

    ax.set_title(models[i])
    ax.legend(loc="lower right")

    if i < 2:
        axins = inset_axes(
            ax,
            width="43%",
            height="50%",
            loc="upper left",
            bbox_to_anchor=(0.2, 0.2, 0.6, 0.6),
            bbox_transform=ax.transAxes,
            borderpad=0,
        )

        axins.plot(fpr, tpr, color="red", lw=2)
        axins.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        axins.set_xlim(-0.01, 0.16)
        axins.set_ylim(0.89, 1.01)

        rect = plt.Rectangle(
            (-0.01, 0.92), 0.17, 0.12, linewidth=1, edgecolor="blue", facecolor="none"
        )
        ax.add_patch(rect)

        arrow_props = dict(
            facecolor="black", edgecolor="black", arrowstyle="<-", mutation_scale=15
        )
        ax.annotate("", xy=(0.075, 0.95), xytext=(0.25, 0.88), arrowprops=arrow_props)

plt.tight_layout()
plt.savefig("results/visualization/roc_cur_aa.png")
plt.savefig("results/visualization/roc_cur_aa.pdf")
