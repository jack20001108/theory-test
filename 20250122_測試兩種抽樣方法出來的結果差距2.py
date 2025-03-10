import numpy as np
from scipy.stats import poisson, kstest
from scipy.spatial import distance_matrix



# 設定參數
lam = 30  # 泊松分布參數
dimensions = 3  # 維度數
sample_sizes = [10, 20, 50, 100]  # 不同樣本數
grid_size = 40  # 用於覆蓋率計算

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
def random_sampling(lam, dimensions, samples):
    return np.random.poisson(lam=lam, size=(samples, dimensions))

# 覆蓋率計算
def calculate_coverage(samples, grid_size):
    max_value = np.max(samples)
    min_value = np.min(samples)
    scaled_samples = (samples - min_value) / (max_value - min_value)
    grid_indices = (scaled_samples * grid_size).astype(int)
    unique_grids = np.unique(grid_indices, axis=0)
    return len(unique_grids) / (grid_size ** samples.shape[1])

# 模擬實驗
for n in sample_sizes:
    lhs_samples = latin_hypercube_sampling(lam, dimensions, n)
    random_samples = random_sampling(lam, dimensions, n)
    
    # 計算覆蓋率
    lhs_coverage = calculate_coverage(lhs_samples, grid_size)
    random_coverage = calculate_coverage(random_samples, grid_size)
    
    # 計算均勻性（使用距離矩陣的均值作為衡量指標）
    lhs_distances = distance_matrix(lhs_samples, lhs_samples).flatten()
    random_distances = distance_matrix(random_samples, random_samples).flatten()
    lhs_uniformity = np.mean(lhs_distances)
    random_uniformity = np.mean(random_distances)
    
    # Kolmogorov-Smirnov 檢驗
    theoretical_dist = poisson.rvs(mu=lam, size=n * dimensions)
    # 使用泊松分布的 CDF 函數
    ks_lhs = kstest(lhs_samples.flatten(), lambda x: poisson.cdf(x, mu=lam))
    ks_random = kstest(random_samples.flatten(), lambda x: poisson.cdf(x, mu=lam))
    

    # 結果輸出
    print(f"樣本數: {n}")
    print(f"LHS 覆蓋率: {lhs_coverage:.4f}, 均勻性: {lhs_uniformity:.4f}, KS p-value: {ks_lhs.pvalue:.4f}")
    print(f"Random 覆蓋率: {random_coverage:.4f}, 均勻性: {random_uniformity:.4f}, KS p-value: {ks_random.pvalue:.4f}")
    print("------")

    import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, kstest

# 設定參數
lam = 30  # 泊松分布參數
dimensions = 3  # 維度數
sample_size = 20  # 樣本數

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
def random_sampling(lam, dimensions, samples):
    return np.random.poisson(lam=lam, size=(samples, dimensions))

# 生成樣本
lhs_samples = latin_hypercube_sampling(lam, dimensions, sample_size)
random_samples = random_sampling(lam, dimensions, sample_size)

# 平坦化樣本
lhs_flattened = lhs_samples.flatten()
random_flattened = random_samples.flatten()

# Kolmogorov-Smirnov 檢驗
ks_lhs = kstest(lhs_flattened, lambda x: poisson.cdf(x, mu=lam))
ks_random = kstest(random_flattened, lambda x: poisson.cdf(x, mu=lam))

# 繪製累積分布函數（CDF）對比圖
x = np.arange(0, 60, 1)  # 定義 x 軸範圍
poisson_cdf = poisson.cdf(x, mu=lam)

plt.figure(figsize=(10, 6))

# 理論泊松分布 CDF
plt.plot(x, poisson_cdf, label="Poisson CDF (Theory)", color="black", linestyle="--")

# LHS 樣本的經驗 CDF
lhs_sorted = np.sort(lhs_flattened)
lhs_ecdf = np.arange(1, len(lhs_sorted) + 1) / len(lhs_sorted)
plt.step(lhs_sorted, lhs_ecdf, label="LHS Empirical CDF", color="blue")

# Random Sampling 的經驗 CDF
random_sorted = np.sort(random_flattened)
random_ecdf = np.arange(1, len(random_sorted) + 1) / len(random_sorted)
plt.step(random_sorted, random_ecdf, label="Random Empirical CDF", color="orange")

# 圖片標註
plt.title("Empirical CDF vs Theoretical Poisson CDF", fontsize=14)
plt.xlabel("Value", fontsize=12)
plt.ylabel("CDF", fontsize=12)
plt.legend(fontsize=10)
plt.grid(True)
plt.show()

# 結果輸出
print(f"LHS Kolmogorov-Smirnov Test: D={ks_lhs.statistic:.4f}, p-value={ks_lhs.pvalue:.4f}")
print(f"Random Kolmogorov-Smirnov Test: D={ks_random.statistic:.4f}, p-value={ks_random.pvalue:.4f}")


