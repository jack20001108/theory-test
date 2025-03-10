import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, wasserstein_distance, ks_2samp
from sklearn.decomposition import PCA
np.set_printoptions(suppress=True, precision=6)

# 設定參數
lambda_poisson = 30
num_products = 3
num_periods = 5
sample_size = 20

# 拉丁超立方抽樣
def latin_hypercube_sampling(lam, dimensions, samples):
    lhs_samples = np.zeros((samples, dimensions))
    for i in range(dimensions):
        cut = np.linspace(0, 1, samples + 1)
        uniform_samples = np.random.uniform(cut[:-1], cut[1:], size=samples)
        np.random.shuffle(uniform_samples)
        poisson_samples = poisson.ppf(uniform_samples, mu=lam).astype(int)
        lhs_samples[:, i] = poisson_samples
    return lhs_samples

# 隨機抽樣
def random_poisson_sampling(lam, dimensions, samples):
    return np.random.poisson(lam=lam, size=(samples, dimensions))

# 覆蓋率檢查
def calculate_coverage(samples, grid_size):
    max_value = np.max(samples)
    min_value = np.min(samples)
    scaled_samples = (samples - min_value) / (max_value - min_value)  # 標準化到 [0, 1]
    grid_indices = (scaled_samples * grid_size).astype(int)
    unique_grids = np.unique(grid_indices, axis=0)
    coverage = len(unique_grids) / (grid_size ** samples.shape[1])
    return coverage

# 高維分布視覺化 (PCA 降維)
def visualize_high_dimensional_distribution(samples, method_name, ax):
    pca = PCA(n_components=2)
    reduced_samples = pca.fit_transform(samples)
    ax.scatter(reduced_samples[:, 0], reduced_samples[:, 1], alpha=0.7, label=method_name)
    ax.set_title(f"{method_name} Distribution (PCA)")
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.legend()

# 主程式
if __name__ == "__main__":
    dimensions = num_products * num_periods

    # 生成樣本
    lhs_samples = latin_hypercube_sampling(lambda_poisson, dimensions, sample_size)
    random_samples = random_poisson_sampling(lambda_poisson, dimensions, sample_size)

    # 計算覆蓋率
    grid_size = 10
    lhs_coverage = calculate_coverage(lhs_samples, grid_size)
    random_coverage = calculate_coverage(random_samples, grid_size)

    # 視覺化分布
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    visualize_high_dimensional_distribution(lhs_samples, "LHS", axs[0])
    visualize_high_dimensional_distribution(random_samples, "Random", axs[1])
    plt.tight_layout()
    plt.show()

    # Wasserstein 距離與理論泊松分布比較
    poisson_theoretical = poisson.rvs(mu=lambda_poisson, size=sample_size * dimensions)
    lhs_flattened = lhs_samples.flatten()
    random_flattened = random_samples.flatten()

    lhs_wasserstein = wasserstein_distance(lhs_flattened, poisson_theoretical)
    random_wasserstein = wasserstein_distance(random_flattened, poisson_theoretical)

    # Kolmogorov-Smirnov 檢驗
    ks_lhs = ks_2samp(lhs_flattened, poisson_theoretical)
    ks_random = ks_2samp(random_flattened, poisson_theoretical)

    # 結果輸出
    print(f"LHS 覆蓋率: {lhs_coverage:.4f}")
    print(f"Random 覆蓋率: {random_coverage:.4f}")

    print(f"Wasserstein Distance (LHS vs Poisson): {lhs_wasserstein:.4f}")
    print(f"Wasserstein Distance (Random vs Poisson): {random_wasserstein:.4f}")

    print(f"Kolmogorov-Smirnov Test (LHS vs Poisson): D={ks_lhs.statistic:.4f}, p-value={ks_lhs.pvalue:.4f}")
    print(f"Kolmogorov-Smirnov Test (Random vs Poisson): D={ks_random.statistic:.4f}, p-value={ks_random.pvalue:.4f}")

    # 判斷結論
    if lhs_coverage > random_coverage:
        print("LHS 的覆蓋率優於 Random。")
    if lhs_wasserstein < random_wasserstein:
        print("LHS 與理論泊松分布的距離小於 Random，分布更接近理論值。")
    if ks_lhs.pvalue > 0.05:
        print("LHS 與理論泊松分布無顯著差異。")
    if ks_random.pvalue <= 0.05:
        print("Random 與理論泊松分布存在顯著差異。")