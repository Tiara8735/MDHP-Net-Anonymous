import numpy as np
from sklearn.metrics import confusion_matrix

from matplotlib import font_manager
from matplotlib import pyplot as plt

# Load Times New Roman font in case it is not installed
font_dirs = ["./fonts/Times New Roman"]
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
for font_file in font_files:
    font_manager.fontManager.addfont(font_file)
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 16

models = ['MDHP-Net', 'SISSA-LSTM', 'SISSA-RNN', 'SISSA-CNN']
file_paths = [
    "results/test/MDHP_NET_N128/statistics",
    "results/test/SISSA_LSTM_N128/statistics",
    "results/test/SISSA_RNN_N128/statistics",
    "results/test/SISSA_CNN_N128/statistics"
]

confusion_matrices = []

for file_path in file_paths:
    true = np.load(file_path + "/labels.npy")
    pred = np.load(file_path + "/pred_labels.npy")
    cm = confusion_matrix(true, pred)
    confusion_matrices.append(cm)

fig, axes = plt.subplots(2, 2, figsize=(10, 6))

plt.subplots_adjust(hspace=0.9)

for i, (cm, model_name) in enumerate(zip(confusion_matrices, models)):
    if i == 4:
        break
    ax = axes[i // 2, i % 2]
    im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['PN', 'PP'], fontsize=16)
    ax.set_yticklabels(['AN', 'AP'], fontsize=16) 

    for m in range(len(cm)):
        for n in range(len(cm[0])):
            color = 'black' if m != n else 'white'
            ax.text(n, m, f'{cm[m, n]}', ha='center', va='center', color=color, fontsize=16)
    
    ax.set_xlabel(f'({chr(97+i)}) {model_name}', fontsize=16)

plt.tight_layout()
plt.savefig("results/visualization/confusion_matrices.png")
plt.savefig("results/visualization/confusion_matrices.pdf")

