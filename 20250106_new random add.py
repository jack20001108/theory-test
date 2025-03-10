import numpy as np
from randomgen import Xoroshiro128

# 動態泊松需求生成函數
def dynamic_poisson_simulation(mean_demand, target_error, num_simulations, max_samples=10000):
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

        # 判斷是否滿足目標誤差
        if standard_error <= target_error:
            break

        # 增加樣本數
        n *= 2  # 每次翻倍增加樣本數

    return np.random.choice(samples, size=num_simulations)  # 返回 num_simulations 個隨機樣本

# 多產品需求生成函數
def generate_multi_product_demand(num_simulations, num_products, num_periods, mean_demands, target_error):
    
    assert len(mean_demands) == num_products, "平均需求長度必須等於產品數量"
    
    demand_matrix = np.zeros((num_simulations, num_products, num_periods))
    
    for product in range(num_products):
        for period in range(num_periods):
            # 為每個產品的每個時間期生成 K 次模擬需求
            demand_values = dynamic_poisson_simulation(
                mean_demand=mean_demands[product],
                target_error=target_error,
                num_simulations=num_simulations
            )
            # 填充需求矩陣
            demand_matrix[:, product, period] = demand_values

    return demand_matrix.astype(int)

# 設定參數
num_simulations = 1000  # 模擬次數 K
num_products = 10       # 產品數量 N
num_periods = 5         # 時間期數 T
mean_demands = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55]  # 每個產品的平均需求 λ
target_error = 0.1      # 目標誤差

# 生成需求矩陣
demand_matrix = generate_multi_product_demand(num_simulations, num_products, num_periods, mean_demands, target_error)

# 輸出結果
print("生成的需求矩陣 (部分)：")
print(demand_matrix[:])  # 顯示前兩組模擬的需求矩陣
