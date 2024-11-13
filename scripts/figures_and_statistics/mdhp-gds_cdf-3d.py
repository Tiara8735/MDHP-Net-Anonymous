import numpy as np
from matplotlib import pyplot as plt
from matplotlib import font_manager
import os
import os.path as osp

# Load Times New Roman font in case it is not installed
font_dirs = ["../fonts/Times New Roman"]
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
for font_file in font_files:
    font_manager.fontManager.addfont(font_file)
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 16

from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize

from mdhpnet.utils.math import compute_cdf_for_each_dim

OUTPUT_DIR = osp.join("results", "visualization")
os.makedirs(OUTPUT_DIR, exist_ok=True)

alpha_path = "./data/rate_0.5_victim_5/train/alpha.npy"
beta_path = "./data/rate_0.5_victim_5/train/beta.npy"
theta_path = "./data/rate_0.5_victim_5/train/theta.npy"
labels_path = "./data/rate_0.5_victim_5/train/labels.npy"

alpha = np.load(alpha_path)  # (n_samples, mdhp_dim, mdhp_dim)
beta = np.load(beta_path)  # (n_samples, mdhp_dim, mdhp_dim)
theta = np.load(theta_path)  # (n_samples, mdhp_dim)
labels = np.load(labels_path)  # (n_samples,)

n_samples, mdhp_dim, _ = alpha.shape

# Chech nan
if np.isnan(alpha).any() or np.isnan(beta).any() or np.isnan(theta).any():
    raise ValueError("NaN detected in the parameters")

# Check inf
if np.isinf(alpha).any() or np.isinf(beta).any() or np.isinf(theta).any():
    raise ValueError("Inf detected in the parameters")


def plot_mdhp_cdf_scatter(
    ax: Axes3D,
    normal_dim_x: np.ndarray,  # (mdhp_dim, n)
    normal_dim_cdf: np.ndarray,  # (mdhp_dim, n)
    attack_dim_x: np.ndarray,  # (mdhp_dim, n)
    attack_dim_cdf: np.ndarray,  # (mdhp_dim, n)
    x_label: str = None,
    dim_label: str = "MDHP Dimension",
    cdf_label: str = "CDF",
):
    normal_color = "blue"
    attack_color = "red"
    scatter_size = 3
    for dim in range(mdhp_dim):
        ax.scatter(
            normal_dim_x[dim],
            np.tile(dim, normal_dim_x.shape[1]),
            normal_dim_cdf[dim],
            color=normal_color,
            s=scatter_size,
        )
        ax.scatter(
            attack_dim_x[dim],
            np.tile(dim, attack_dim_x.shape[1]),
            attack_dim_cdf[dim],
            color=attack_color,
            s=scatter_size,
        )
    ax.set_xlabel(x_label)
    ax.set_ylabel(dim_label)
    ax.set_zlabel(cdf_label)


def calculate_x_dim_cdf_grids(
    dim: int,
    dim_x: np.ndarray,
    dim_cdf: np.ndarray,
    n_x_linspace: int = 100,
    interp_method: str = "linear",
):
    x_linspace = np.linspace(dim_x.min(), dim_x.max(), n_x_linspace)
    dim_linspace = np.linspace(0, dim - 1, dim)
    x_grid, dim_grid = np.meshgrid(x_linspace, dim_linspace)
    cdf_grid = griddata(
        points=(
            dim_x.flatten(),
            np.tile(np.arange(mdhp_dim), dim_x.shape[1]),
        ),
        values=dim_cdf.flatten(),
        xi=(x_grid, dim_grid),
        method=interp_method,
    )
    # Replace nan with 1.0
    cdf_grid = np.nan_to_num(cdf_grid, nan=1.0)
    return x_grid, dim_grid, cdf_grid


def plot_mdhp_cdf_surface(
    ax: Axes3D,
    normal_x_grid: np.ndarray,
    normal_dim_grid: np.ndarray,
    normal_cdf_grid: np.ndarray,
    normal_transparency: float,
    attack_x_grid: np.ndarray,
    attack_dim_grid: np.ndarray,
    attack_cdf_grid: np.ndarray,
    attack_transparency: float,
    x_label: str,
    dim_label: str = "MDHP Dimension",
    cdf_label: str = "CDF",
):
    # Normalize the CDF values to use them in custom colormap
    norm1 = Normalize(vmin=normal_cdf_grid.min(), vmax=normal_cdf_grid.max())
    norm2 = Normalize(vmin=attack_cdf_grid.min(), vmax=attack_cdf_grid.max())
    # Get colormap
    cmap1 = plt.get_cmap("viridis")
    cmap2 = plt.get_cmap("plasma")
    # Apply colormap and add transparency by setting the alpha channel
    facecolors1 = cmap1(norm1(normal_cdf_grid))
    facecolors1[..., 3] = normal_transparency  # Set transparency for the first surface
    facecolors2 = cmap2(norm2(attack_cdf_grid))
    facecolors2[..., 3] = attack_transparency  # Set transparency for the second surface

    ax.plot_surface(
        normal_x_grid,
        normal_dim_grid,
        normal_cdf_grid,
        facecolors=facecolors1,
        shade=False,
    )
    ax.plot_surface(
        attack_x_grid,
        attack_dim_grid,
        attack_cdf_grid,
        facecolors=facecolors2,
        shade=False,
    )

    ax.set_xlabel(x_label)
    ax.set_ylabel(dim_label)
    ax.set_zlabel(cdf_label)


target_dim = 5

fig = plt.figure(figsize=(12, 6))

# Ax1: alpha ============================================================================

ax1: Axes3D = fig.add_subplot(121, projection="3d")

# Shape: (n_samples, mdhp_dim)
target_alpha_normal_all_dims = alpha[labels == 0][:, target_dim, :]
# Shape: (n_samples, mdhp_dim)
target_alpha_attack_all_dims = alpha[labels == 1][:, target_dim, :]

# Shape: ((mdhp_dim, n_samples), (mdhp_dim, n_samples))
alpha_normal_dim_x, alpha_normal_dim_cdf = compute_cdf_for_each_dim(
    target_alpha_normal_all_dims
)
alpha_normal_x_grid, alpha_normal_dim_grid, alpha_normal_cdf_grid = (
    calculate_x_dim_cdf_grids(mdhp_dim, alpha_normal_dim_x, alpha_normal_dim_cdf, 100)
)

# Shape: ((mdhp_dim, n_samples), (mdhp_dim, n_samples))
alpha_attack_dim_x, alpha_attack_dim_cdf = compute_cdf_for_each_dim(
    target_alpha_attack_all_dims
)
alpha_attack_x_grid, alpha_attack_dim_grid, alpha_attack_cdf_grid = (
    calculate_x_dim_cdf_grids(mdhp_dim, alpha_attack_dim_x, alpha_attack_dim_cdf, 100)
)

plot_mdhp_cdf_surface(
    ax1,
    alpha_normal_x_grid,
    alpha_normal_dim_grid,
    alpha_normal_cdf_grid,
    0.6,
    alpha_attack_x_grid,
    alpha_attack_dim_grid,
    alpha_attack_cdf_grid,
    0.4,
    x_label=r"$\alpha$",
    dim_label="MDHP Dimension",
    cdf_label=None,
)

# Ax2: beta =============================================================================

ax2: Axes3D = fig.add_subplot(122, projection="3d")

# Shape: (n_samples, mdhp_dim)
target_beta_normal_all_dims = beta[labels == 0][:, target_dim, :]
# Shape: (n_samples, mdhp_dim)
target_beta_attack_all_dims = beta[labels == 1][:, target_dim, :]

# Shape: ((mdhp_dim, n_samples), (mdhp_dim, n_samples))
beta_normal_dim_x, beta_normal_dim_cdf = compute_cdf_for_each_dim(
    target_beta_normal_all_dims
)

# Shape: ((mdhp_dim, n_samples), (mdhp_dim, n_samples))
beta_attack_dim_x, beta_attack_dim_cdf = compute_cdf_for_each_dim(
    target_beta_attack_all_dims
)

plot_mdhp_cdf_scatter(
    ax2,
    beta_normal_dim_x,
    beta_normal_dim_cdf,
    beta_attack_dim_x,
    beta_attack_dim_cdf,
    x_label=r"$\beta$",
    dim_label="MDHP Dimension",
    cdf_label=None,
)

plt.savefig(osp.join(OUTPUT_DIR, f"mdhp-gds_cdf-{target_dim}-3d.jpg"))
plt.savefig(osp.join(OUTPUT_DIR, f"mdhp-gds_cdf-{target_dim}-3d.pdf"))
