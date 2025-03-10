import numpy as np
from scipy.stats import norm

def fitness(order, leadtime, begin_inventory, bom, demands):

    # 呼叫 evaluator 函數

    #inventory,shortage,reshape_unclaim,shortage_com,safe_aligned_bom_demand,shortage_com_tr = evaluator(demands, order, leadtime, begin_inventory, bom)


    #shortage_com_tr
    # 計算 fill rate
    #com_fill_rate = 1 - (shortage_com / safe_aligned_bom_demand)

    # 計算當前樣本的產品填充率
    #total_demand = np.sum(demands, axis=1)  # 每個產品的總需求
    #total_supply = np.maximum(total_demand - np.sum(shortage, axis=1), 0)  # 每個產品的滿足需求數量
    #product_fill_rate = total_supply / total_demand  # 每個產品的填充率

    print("com_fill_rate")
    print(com_fill_rate)
    # 返回當前樣本的平均產品填充率
    #return np.mean(product_fill_rate)
    return(inventory)


def latin_hypercube_sampling(dimensions, samples):

    result = np.zeros((samples, dimensions))
    for i in range(dimensions):
        cut = np.linspace(0, 1, samples + 1)  # 分層切割
        uniform_samples = np.random.uniform(cut[:-1], cut[1:], size=samples) #每個子區間的均勻隨機樣本
        np.random.shuffle(uniform_samples) # 打散隨機樣本
        result[:, i] = uniform_samples # 填入結果
    return result

def simulate_with_dynamic_samples(
    num_products, initial_samples=50,
    sample_increment=50, max_iterations=10, target_precision=0.01,
    order=None, leadtime=None, begin_inventory=None, bom=None, num_periods=5
):
    total_samples = initial_samples
    cumulative_fill_rate = []  # 用於記錄累積平均填充率
    previous_cumulative_mean = None  # 前一次的累積平均值

    for iteration in range(max_iterations):
        
        # 使用拉丁超立方抽樣產生樣本
        lhs_samples = latin_hypercube_sampling(num_products * num_periods, total_samples)

        # 將樣本映射到整數範圍
        min_demand = 10  # 需求的最小值
        max_demand = 50  # 需求的最大值
        demand_samples = np.floor(lhs_samples * (max_demand - min_demand + 1) + min_demand).astype(int)

        print("demand_samples")
        print(demand_samples)
        # 確保形狀正確
        demand_samples = demand_samples.reshape(total_samples, num_products, num_periods)
        print("demand_samples reshape")
        print(demand_samples)
        # 記錄當前樣本的填充率
        fill_rates = []  # 當前樣本的填充率
        for demand in demand_samples:
            # 每組樣本分別傳入 fitness，demand 的形狀為 (num_products, num_periods)
            fill_rate = fitness(
                demands=demand,
                order=order,
                leadtime=leadtime,
                begin_inventory=begin_inventory,
                bom=bom
            )
            fill_rates.append(fill_rate)

            # 動態計算累積平均填充率
            cumulative_fill_rate.append(fill_rate)
            cumulative_mean = np.mean(cumulative_fill_rate)

            # 即時檢查收斂性
            if previous_cumulative_mean is not None:
                relative_error = np.abs(cumulative_mean - previous_cumulative_mean) / cumulative_mean
                print(f"第 {iteration + 1} 次迭代: 累積平均填充率 = {cumulative_mean:.4f}, 相對誤差 = {relative_error:.4f}")

                if relative_error < target_precision:
                    print(f"結果收斂於第 {iteration + 1} 次迭代")
                    return cumulative_fill_rate, cumulative_mean, len(cumulative_fill_rate)

            # 更新前一次累積平均值
            previous_cumulative_mean = cumulative_mean

        # 增加樣本數
        total_samples += sample_increment

    return cumulative_fill_rate, cumulative_mean, len(cumulative_fill_rate)
