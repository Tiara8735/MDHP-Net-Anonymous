import numpy as np
from typing import List, Tuple
from os import path as osp
from easydict import EasyDict as edict
from ruamel.yaml import YAML

from matplotlib import font_manager
from matplotlib import pyplot as plt

# Load Times New Roman font in case it is not installed
font_dirs = ["./fonts/Times New Roman"]
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
for font_file in font_files:
    font_manager.fontManager.addfont(font_file)
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 16

from mdhpnet.utils.math import accuracy_score, precision_score, recall_score, f1_score


def cal_stats_all_n_seq(
    dir_list: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    accs, precs, recalls, f1s = [], [], [], []
    for dir in dir_list:
        gt_labels = np.load(osp.join(dir, "labels.npy"))
        pred_labels = np.load(osp.join(dir, "pred_labels.npy"))
        accs.append(accuracy_score(gt_labels, pred_labels))
        precs.append(precision_score(gt_labels, pred_labels))
        recalls.append(recall_score(gt_labels, pred_labels))
        f1s.append(f1_score(gt_labels, pred_labels))
    return np.array(accs), np.array(precs), np.array(recalls), np.array(f1s)


def plot_ys_same_x(
    ax: plt.Axes,
    x: np.ndarray,
    ys: List[np.ndarray],
    labels: List[str],
    y_name: str,
    x_name: str = "Num of Seq Per Window",
):
    for y, label in zip(ys, labels):
        ax.plot(x, y, label=label, linewidth=2, marker='x', markersize=10)
    ax.legend(loc='lower right')
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.grid(True)
    ax.set_xticks(x)



def main(cfg: dict | edict):
    cfg = edict(cfg)

    # Y-axis
    mdhp_net_stats = cal_stats_all_n_seq(cfg.MDHP_NET_DIRS)
    sissa_lstm_stats = cal_stats_all_n_seq(cfg.SISSA_LSTM_DIRS)
    sissa_rnn_stats = cal_stats_all_n_seq(cfg.SISSA_RNN_DIRS)
    sissa_cnn_stats = cal_stats_all_n_seq(cfg.SISSA_CNN_DIRS)

    # X-axis
    x_n_seqs = [32, 48, 64, 80, 96, 112, 128]
    labels = ["MDHP-Net", "SISSA-LSTM", "SISSA-RNN", "SISSA-CNN"]

    fig = plt.figure(figsize=(12, 10))

    # Fig Accuracy: 2 rows, 2 columns, 1st subplot
    ax1 = fig.add_subplot(221)
    plot_ys_same_x(
        ax=ax1,
        x=x_n_seqs,
        ys=[
            mdhp_net_stats[0],
            sissa_lstm_stats[0],
            sissa_rnn_stats[0],
            sissa_cnn_stats[0],
        ],
        labels=labels,
        y_name="Accuracy",
    )

    # Fig Precision: 2 rows, 2 columns, 2nd subplot
    ax2 = fig.add_subplot(222)
    plot_ys_same_x(
        ax=ax2,
        x=x_n_seqs,
        ys=[
            mdhp_net_stats[1],
            sissa_lstm_stats[1],
            sissa_rnn_stats[1],
            sissa_cnn_stats[1],
        ],
        labels=labels,
        y_name="Precision",
    )

    # Fig Recall: 2 rows, 2 colums, 3th subplot
    ax3 = fig.add_subplot(223)
    plot_ys_same_x(
        ax=ax3,
        x=x_n_seqs,
        ys=[
            mdhp_net_stats[2],
            sissa_lstm_stats[2],
            sissa_rnn_stats[2],
            sissa_cnn_stats[2],
        ],
        labels=labels,
        y_name="Recall",
    )

    # Fig F1 Score: 2 rows, 2 colums, 4th subplot
    ax4 = fig.add_subplot(224)
    plot_ys_same_x(
        ax=ax4,
        x=x_n_seqs,
        ys=[
            mdhp_net_stats[3],
            sissa_lstm_stats[3],
            sissa_rnn_stats[3],
            sissa_cnn_stats[3],
        ],
        labels=labels,
        y_name="F1 Score",
    )

    plt.savefig(
        "results/visualization/n_seq_comp.jpg",
        bbox_inches="tight",
        pad_inches=0.1,
    )

    plt.savefig(
        "results/visualization/n_seq_comp.pdf",
        bbox_inches="tight",
        pad_inches=0.1,
    )


if __name__ == "__main__":
    cfg = YAML().load(open("configs/visualization/n_seq_comp.yml"))
    main(cfg=cfg)
