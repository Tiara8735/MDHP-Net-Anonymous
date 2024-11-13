import numpy as np
from typing import List
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


def str2lineStyle(style: str) -> str:
    if style in ["solid", "-"]:
        return "-"
    elif style in ["dashed", "--"]:
        return "--"
    elif style in ["dashdot", "-."]:
        return "-."
    elif style in ["dotted", ":"]:
        return ":"
    else:
        raise ValueError("Invalid line style")


def plot_loss_acc(
    ax: plt.Axes,
    lines: List[dict],
    accYLabel: bool = True,
    lossYLabel: bool = True,
    legendLoc: str = "center right",
):
    axLoss = ax
    axLoss.set_ylim(0, 1)
    if lossYLabel:
        axLoss.set_ylabel("Loss")
    axAcc = axLoss.twinx()
    axAcc.set_ylim(0.7, 1)
    if accYLabel:
        axAcc.set_ylabel("Accuracy")
    x = None

    allHandles = []

    for line in lines:
        # Load data
        data = np.load(file=line["dataPath"])

        # loss line -> axLoss
        if line["type"] == "loss":
            ax = axLoss
            ax.set_ylim(0, np.max(data) * 1.1)
        # acc line -> axAcc
        elif line["type"] == "acc":
            ax = axAcc
        else:
            raise ValueError("Invalid type")

        if x is None:
            x = np.arange(len(data))
        # Each line should have the same x length
        assert len(x) == len(data)

        # Plot the line
        (handle,) = ax.plot(
            x,
            data,
            color=line["hexColor"],
            label=line["label"],
            linestyle=str2lineStyle(line["lineStyle"]),
        )

        allHandles.append(handle)

    # Create legends for both axLoss and axAcc
    # handles1, labels1 = axLoss.get_legend_handles_labels()
    # handles2, labels2 = axAcc.get_legend_handles_labels()
    axLoss.set_xlabel("Epoch")
    # Combine the handles and labels for the legend
    axAcc.legend(handles=allHandles, loc=legendLoc)


def main():
    fig = plt.figure(figsize=(24, 5.5))
    plt.subplots_adjust(hspace=0.23)

    ax = fig.add_subplot(131)
    config = YAML().load(
        open("configs/visualization/train_mdhp-net_vs_sissa-lstm.yml", "r")
    )
    plot_loss_acc(ax, config["Lines"], accYLabel=False)
    ax.text(0.5, -0.15, "(a)", ha="center", va="top", transform=ax.transAxes)
    
    ax = fig.add_subplot(132)
    config = YAML().load(
        open("configs/visualization/train_mdhp-net_vs_sissa-rnn.yml", "r")
    )

    plot_loss_acc(ax, config["Lines"], lossYLabel=False, accYLabel=False)
    ax.text(0.5, -0.15, "(b)", ha="center", va="top", transform=ax.transAxes)

    ax = fig.add_subplot(133)
    config = YAML().load(
        open("configs/visualization/train_mdhp-net_vs_sissa-cnn.yml", "r")
    )
    plot_loss_acc(ax, config["Lines"], lossYLabel=False)
    ax.text(0.5, -0.15, "(c)", ha="center", va="top", transform=ax.transAxes)

    plt.savefig(
        "results/visualization/acc-loss.jpg",
        bbox_inches="tight",
        pad_inches=0.1,
    )

    plt.savefig(
        "results/visualization/acc-loss.pdf",
        bbox_inches="tight",
        pad_inches=0.1,
    )


if __name__ == "__main__":
    main()
