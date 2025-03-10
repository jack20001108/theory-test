import numpy as np
from pyDOE2 import lhs

def latin_hypercube_sampling(mean_demand, num_simulations):
    """
    使用拉丁超立方抽樣生成泊松分布的樣本
    """
    num_bins = 1000  # 定義分層數
    uniform_samples = lhs(1, samples=num_simulations, criterion='maximin')  # 均勻分布樣本
    
    # 生成泊松分布樣本
    poisson_samples = np.random.poisson(lam=mean_demand, size=num_bins)
    
    # 將均勻樣本映射到泊松分布
    bin_indices = (uniform_samples * num_bins).astype(int).flatten()
    sampled_values = poisson_samples[bin_indices]
    
    return sampled_values

def generate_multi_product_demand_lhs(num_simulations, num_products, num_periods, mean_demands):
    """
    基於拉丁超立方抽樣生成三種產品、五個需求期間的需求矩陣
    """
    assert len(mean_demands) == num_products, "平均需求長度必須等於產品數量"

    # 初始化需求矩陣
    demand_matrix = np.zeros((num_simulations, num_products, num_periods))

    # 為每種產品和每個需求期間生成需求
    for product in range(num_products):
        for period in range(num_periods):
            # 拉丁超立方抽樣，對應於當前產品和時間期
            demand_values = latin_hypercube_sampling(
                mean_demand=mean_demands[product],
                num_simulations=num_simulations
            )
            # 將生成的需求填入需求矩陣
            demand_matrix[:, product, period] = demand_values

        print(f"產品 {product+1} 的需求抽樣完成")
    
    return demand_matrix.astype(int)

# 測試參數
num_simulations = 1000  # 模擬次數 K
num_products = 3       # 產品數量 N
num_periods = 5         # 時間期數 T
mean_demands = [10, 30, 55]  # 每種產品的平均需求

# 每次執行時，生成獨立的需求矩陣
demand_matrix_1 = generate_multi_product_demand_lhs(num_simulations, num_products, num_periods, mean_demands)
demand_matrix_2 = generate_multi_product_demand_lhs(num_simulations, num_products, num_periods, mean_demands)

# 確認結果不同
print("第一次生成的需求矩陣：")
print(demand_matrix_1)
print("第二次生成的需求矩陣：")
print(demand_matrix_2)

# 確認需求矩陣是否不同
print("需求矩陣是否相同：", np.array_equal(demand_matrix_1, demand_matrix_2))
