from matplotlib import font_manager
from matplotlib import pyplot as plt

# Load Times New Roman font in case it is not installed
font_dirs = ["./fonts/Times New Roman"]
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
for font_file in font_files:
    font_manager.fontManager.addfont(font_file)
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.style"] = "normal"
plt.rcParams["font.size"] = 16

import numpy as np
from os import path as osp

from mdhpnet.utils.math import compute_cdf


def plot_cdf(ax: plt.Axes, x_normal, y_normal, x_attack, y_attack, xlabel, ylabel=""):
    ax.plot(x_normal, y_normal, color="blue", label="Normal", linewidth=2)
    ax.plot(x_attack, y_attack, color="red", label="Attack", linewidth=2)
    # ax.scatter(x_normal, y_normal, color="blue", s=5)
    # ax.scatter(x_attack, y_attack, color="red", s=5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()


def main():
    # Select Dim
    # TARGET_DIMS = [(3, 0), (3, 1), (3, 2), (4, 3), (4, 4), (4, 5)]
    TARGET_DIMS = [(3, 5), (5, 5)]

    # Load data
    DATA_DIR = osp.join("data", "rate_0.4_victim_4", "train")
    ALPHA_PATH = osp.join(DATA_DIR, "alpha.npy")
    BETA_PATH = osp.join(DATA_DIR, "beta.npy")
    THETA_PATH = osp.join(DATA_DIR, "theta.npy")
    LABELS_PATH = osp.join(DATA_DIR, "labels.npy")
    ALPHA = np.load(ALPHA_PATH)
    BETA = np.load(BETA_PATH)
    THETA = np.load(THETA_PATH)
    LABELS = np.load(LABELS_PATH)

    NUM_ROW = len(TARGET_DIMS)

    # Create figure
    fig = plt.figure(figsize=(13, 3.5 * NUM_ROW))
    # Set the row spacing for each subplot
    # fig.subplots_adjust(hspace=2.0)

    for i, target_dim in enumerate(TARGET_DIMS):

        ALPHA_NORMAL = ALPHA[LABELS == 0][:, target_dim[0], target_dim[1]]
        BETA_NORMAL = BETA[LABELS == 0][:, target_dim[0], target_dim[1]]
        THETA_NORMAL = THETA[LABELS == 0][:, target_dim[0]]

        ALPHA_ATTACK = ALPHA[LABELS == 1][:, target_dim[0], target_dim[1]]
        BETA_ATTACK = BETA[LABELS == 1][:, target_dim[0], target_dim[1]]
        THETA_ATTACK = THETA[LABELS == 1][:, target_dim[0]]

        ax1 = fig.add_subplot(NUM_ROW, 3, 3 * i + 1)
        alpha_x_normal, alpha_y_normal = compute_cdf(ALPHA_NORMAL)
        alpha_x_attack, alpha_y_attack = compute_cdf(ALPHA_ATTACK)

        plot_cdf(
            ax1,
            alpha_x_normal,
            alpha_y_normal,
            alpha_x_attack,
            alpha_y_attack,
            r"$\alpha$" if i == len(TARGET_DIMS) - 1 else "",
            f"CDF",
        )

        ax2 = fig.add_subplot(NUM_ROW, 3, 3 * i + 2)
        beta_x_normal, beta_y_normal = compute_cdf(BETA_NORMAL)
        beta_x_attack, beta_y_attack = compute_cdf(BETA_ATTACK)
        plot_cdf(
            ax2,
            beta_x_normal,
            beta_y_normal,
            beta_x_attack,
            beta_y_attack,
            r"$\beta$" if i == len(TARGET_DIMS) - 1 else "",
        )

        ax3 = fig.add_subplot(NUM_ROW, 3, 3 * i + 3)
        theta_x_normal, theta_y_normal = compute_cdf(THETA_NORMAL)
        theta_x_attack, theta_y_attack = compute_cdf(THETA_ATTACK)
        plot_cdf(
            ax3,
            theta_x_normal,
            theta_y_normal,
            theta_x_attack,
            theta_y_attack,
            r"$\theta$" if i == len(TARGET_DIMS) - 1 else "",
        )

        row_label = f"({chr(97 + i)}) Dim {target_dim[0]}-{target_dim[1]}"
        y_pos = 1 - (i + 0.95) * (1.0 / NUM_ROW)
        fig.text(0.5, y_pos, row_label, fontsize=14, va="top", ha="center")

    fig.tight_layout()

    plt.savefig(
        f"results/visualization/cdf-abt-2d.pdf",
        bbox_inches="tight",
        pad_inches=0.1,
    )

    plt.savefig(
        f"results/visualization/cdf-abt-2d.jpg",
        bbox_inches="tight",
        pad_inches=0.1,
    )


if __name__ == "__main__":
    main()
