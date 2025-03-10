import numpy as np
from randomgen import Xoroshiro128

def dynamic_poisson_simulation(mean_demand, target_error, max_samples=10000):
    """
    動態調整樣本數的泊松分佈模擬。
    
    Args:
        mean_demand (float): 泊松分佈的平均需求量 λ。
        target_error (float): 目標誤差（如 ±0.05 表示 ±5%）。
        max_samples (int): 最大樣本數。
    
    Returns:
        float: 最終估計均值。
        float: 最終誤差。
        int: 使用的樣本數。
    """
    rng = np.random.Generator(Xoroshiro128(seed=42))
    samples = []
    n = 100  # 初始樣本數

    while n <= max_samples:
        # 生成新樣本
        new_samples = rng.poisson(lam=mean_demand, size=n)
        samples.extend(new_samples)
        
        # 計算樣本均值和標準誤差
        sample_mean = np.mean(samples)
        sample_std_dev = np.std(samples, ddof=1)
        standard_error = 1.96 * sample_std_dev / np.sqrt(len(samples))
        print(samples)
        print(f"樣本數: {len(samples)}, 平均值: {sample_mean:.2f}, 誤差範圍: ±{standard_error:.2f}")
        
        # 判斷是否滿足目標誤差
        if standard_error <= target_error:
            break
        
        # 增加樣本數
        n *= 2  # 每次翻倍增加樣本數
    
    return sample_mean, standard_error, len(samples)

# 執行模擬
result = dynamic_poisson_simulation(mean_demand=10, target_error=0.1)  # 誤差目標 ±5%
print(f"模擬完成：平均值 = {result[0]:.2f}, 最終誤差 = ±{result[1]:.2f}, 總樣本數 = {result[2]}")
