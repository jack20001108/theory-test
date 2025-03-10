
import numpy as np
from scipy.stats import norm
np.set_printoptions(suppress=True, precision=6)
import time
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
from functools import partial
from scipy.stats import skew, kurtosis, shapiro,poisson
from scipy.stats import kstest, norm


'''=================設定參數========================='''

num_products = 3
num_materials = 4
num_periods = 5  
leadtime = 1
inventory_cost_per_unit = 2
num_generations = 300
target_fill_rate = 0.95


bom = np.array([
    [1, 0, 1],
    [2, 1, 0],
    [0, 1, 1],
    [1, 1, 0]
])

order = np.array([
    [25, 21, 0, 0, 0, 0],
    [30, 0, 35, 0, 0, 0],
    [40, 0, 0, 28, 0, 0],
    [38, 0, 0, 0, 32, 0]
])

'''==================主程式=========================='''

# 輸入一組需求與訂單，輸出缺貨情況(缺貨時間,缺貨數量) 
def evaluator(demand, order, leadtime):  
    num_product,num_period = demand.shape   # (產品數量,期數)
    inventory = np.zeros((num_materials, num_period))   # 庫存(M*T+1)

    '----計算庫存----'
    # 計算庫存要用到的數據
    begin_inventory = np.tile(order[:, 0][:, None], (1, num_periods))   # 初始庫存--> order第一列,展開成(M*T)
    
    bom_demand = bom[:, :, None] * demand   # 零件需求展開(N,M,T)
    material_demand = bom_demand.transpose(2,0,1)   # 每個產品的需求(T,N,M)
    cumulative_demand = np.cumsum(bom@demand,axis=1)    # 累積需求(N,T)
    
    real_order = order[:, 1:]   # 每期訂貨量--> order第二列到最後一列
    arrival_order=np.roll(real_order, shift=leadtime, axis=1)   # order往後推leadtime期到達
    arrival_order[:,:leadtime]=0    # 在前leadtime期沒有到貨
    cumulative_arrival=np.cumsum(arrival_order, axis=1)  # 累積到貨

    # 計算庫存
    inventory=begin_inventory+cumulative_arrival-cumulative_demand
    real_stock=np.where(inventory<0,0,inventory) # 將庫存<0的數值取0

    '----計算未分配需求----'
    # 計算未分配需求要用到的數據
    shifted_bom_demand = np.roll(bom_demand, shift=-1, axis=1)  # 右移一格
    shifted_bom_demand[:, -1, :] = 0  # 將最後一列補0
    material_reverse_cumulative_demand =np.cumsum(shifted_bom_demand[:, ::-1, :],axis=1)[:, ::-1, :] # 1.反轉時間軸累加 2.再反轉回來
    
    changed_inventory=inventory[:, :,None] 
    material_period_end_inventory=changed_inventory.transpose(0,2,1) # 使庫存與未分配需求形狀相符

    # 計算未分配零件數量
    unclaim_inventory = (material_period_end_inventory + material_reverse_cumulative_demand) 
    material_unclaim=unclaim_inventory.transpose(2,0,1) # 每個產品的未分配需求
        # product_shortage=unclaim_inventory.transpose(1,0,2) # 各產品所需的零件，在各期間是否缺貨 

    return material_demand,real_stock,material_unclaim

# 輸入一組訂單，輸出平均滿足率與存貨量
def fitness_lhs(order):
    '----生成隨機需求樣本----'
    # 設定初始樣本參數
    
    initial_sample_size = 50  # 初始樣本數
    max_sample_size = 2500   # 最大樣本數，防止無限循環
    epsilon = 0.01            # 收斂條件：均值與期望值的差距
    lam = 30                  # 泊松分布的參數（期望值）
    sigma_threshold = np.sqrt(lam)    # 收斂條件：標準差閾值
    normality_threshold = 0.05  # 常態性檢定 p-value 閾值

    sample_size = initial_sample_size
    converged = False



    while not converged:

        # 生成 LHS 樣本
        lhs_samples = latin_hypercube_sampling(lam,num_products * num_periods, sample_size)
        #random_samples = random_sampling(lam,num_products * num_periods, sample_size)
        demand_samples = lhs_samples.reshape(sample_size, num_products, num_periods)

        # 計算樣本均值和標準差
        sample_mean = np.mean(demand_samples)
        sample_std = np.std(demand_samples)
        skewness =  skew(demand_samples)
        kurt = kurtosis(demand_samples)
        # 改用 Kolmogorov-Smirnov 檢驗
        ks_statistic, ks_p_value = kstest(demand_samples.flatten(), 'norm', args=(sample_mean, sample_std))

        print(f"樣本數: {sample_size}, 均值: {sample_mean:.2f}, 標準差: {sample_std:.2f}")
        print(f"Kolmogorov-Smirnov 檢驗 p-value: {ks_p_value:.4f}")



        # 檢查收斂條件
        if abs(sample_mean - lam) < epsilon and abs(sample_std - sigma_threshold) < epsilon and ks_p_value < normality_threshold:
            converged = True
        else:
            sample_size += 50  # 增加樣本數
            if sample_size > max_sample_size:
                print("樣本數達到上限，未能滿足收斂條件")
                break

    '----計算平均滿足率----'
    cumulative_fill_rate = []
    real_stock_list = []

    # 計算平均滿足率
    for demand in demand_samples:
        # 呼叫evaluator
        material_demand, real_stock, material_unclaim = evaluator(demand, order, leadtime)

        # 計算滿足率
        shortage_com = np.where(material_unclaim < 0, np.minimum(-material_unclaim, material_demand), 0)  # 計算零件缺貨量
        safe_material_demand = np.where(material_demand == 0, 1e-10, material_demand)  # 避免分母為零
        com_fill_rate = 1 - (shortage_com / safe_material_demand)  # 零件滿足率

        com_fill_rate_tr = com_fill_rate.transpose(2, 1, 0)  # 轉置為(T,N,M)
        pro_fill_rate = np.min(com_fill_rate_tr, axis=1)  # 產品滿足率
        cumulative_fill_rate.append(pro_fill_rate)  # 累積產品滿足率

        # 計算存貨量
        real_stock_list.append(real_stock)

    # 計算累積平均產品滿足率
    cumulative_fill_rate_array = np.array(cumulative_fill_rate)
    cumulative_fill_rate_mean = np.mean(cumulative_fill_rate_array, axis=0)
    # 計算累積存貨量
    real_stock_array = np.array(real_stock_list)
    real_stock_mean = np.mean(real_stock_array, axis=0)

    return cumulative_fill_rate_mean, real_stock_mean, sample_size



# 輸入一組訂單，輸出平均滿足率與存貨量
def fitness_rs(order):
    '----生成隨機需求樣本----'
    # 設定初始樣本參數
    
    initial_sample_size = 50  # 初始樣本數
    max_sample_size = 2500   # 最大樣本數，防止無限循環
    epsilon = 0.01            # 收斂條件：均值與期望值的差距
    lam = 30                  # 泊松分布的參數（期望值）
    sigma_threshold = np.sqrt(lam)    # 收斂條件：標準差閾值
    normality_threshold = 0.05  # 常態性檢定 p-value 閾值

    sample_size = initial_sample_size
    converged = False

    while not converged:

        # 生成 LHS 樣本
        #lhs_samples = latin_hypercube_sampling(lam,num_products * num_periods, sample_size)
        random_samples = random_sampling(lam,num_products * num_periods, sample_size)
        demand_samples = random_samples.reshape(sample_size, num_products, num_periods)

        # 計算樣本均值和標準差
        sample_mean = np.mean(demand_samples)
        sample_std = np.std(demand_samples)
        skewness =  skew(demand_samples)
        kurt = kurtosis(demand_samples)
        # 改用 Kolmogorov-Smirnov 檢驗
        ks_statistic, ks_p_value = kstest(demand_samples.flatten(), 'norm', args=(sample_mean, sample_std))

        print(f"樣本數: {sample_size}, 均值: {sample_mean:.2f}, 標準差: {sample_std:.2f}")
        print(f"Kolmogorov-Smirnov 檢驗 p-value: {ks_p_value:.4f}")

        # 檢查收斂條件
        if abs(sample_mean - lam) < epsilon and abs(sample_std - sigma_threshold) < epsilon and ks_p_value < normality_threshold and sample_size > 400:
            converged = True
        else:
            sample_size += 50  # 增加樣本數
            if sample_size > max_sample_size:
                print("樣本數達到上限，未能滿足收斂條件")
                break

    '----計算平均滿足率----'
    cumulative_fill_rate = []
    real_stock_list = []

    # 計算平均滿足率
    for demand in demand_samples:
        # 呼叫evaluator
        material_demand, real_stock, material_unclaim = evaluator(demand, order, leadtime)

        # 計算滿足率
        shortage_com = np.where(material_unclaim < 0, np.minimum(-material_unclaim, material_demand), 0)  # 計算零件缺貨量
        safe_material_demand = np.where(material_demand == 0, 1e-10, material_demand)  # 避免分母為零
        com_fill_rate = 1 - (shortage_com / safe_material_demand)  # 零件滿足率

        com_fill_rate_tr = com_fill_rate.transpose(2, 1, 0)  # 轉置為(T,N,M)
        pro_fill_rate = np.min(com_fill_rate_tr, axis=1)  # 產品滿足率
        cumulative_fill_rate.append(pro_fill_rate)  # 累積產品滿足率

        # 計算存貨量
        real_stock_list.append(real_stock)

    # 計算累積平均產品滿足率
    cumulative_fill_rate_array = np.array(cumulative_fill_rate)
    cumulative_fill_rate_mean = np.mean(cumulative_fill_rate_array, axis=0)
    # 計算累積存貨量
    real_stock_array = np.array(real_stock_list)
    real_stock_mean = np.mean(real_stock_array, axis=0)

    return cumulative_fill_rate_mean, real_stock_mean, sample_size

# 輸入生成維度、樣本數，輸出LHS矩陣
def latin_hypercube_sampling(lam,dimensions, samples): 
    lhs_samples = np.zeros((samples, dimensions))

    for i in range(dimensions):
        cut = np.linspace(0, 1, samples + 1)  #切割成樣本數量的空間
        uniform_samples = np.random.uniform(cut[:-1], cut[1:], size=samples) #每個區間產生均勻隨機樣本
        np.random.shuffle(uniform_samples) #順序隨機打散
        # 利用泊松分布的 PPF (Percent-Point Function) 映射均勻樣本到泊松分布
        poisson_samples = poisson.ppf(uniform_samples, mu=lam).astype(int)
        
        # 將映射後的樣本存入 LHS 矩陣
        lhs_samples[:, i] = poisson_samples
    return lhs_samples

def random_sampling(lam, dimensions, samples):
    return np.random.poisson(lam=lam, size=(samples, dimensions))



def visualize_combined_mean_convergence(order, lam, sampling_functions, labels):
    sample_sizes = {label: [] for label in labels}
    means = {label: [] for label in labels}


    initial_sample_size = 50  # 初始樣本數
    max_sample_size = 2500   # 最大樣本數，防止無限循環
    epsilon = 0.01            # 收斂條件：均值與期望值的差距
    lam = 30                  # 泊松分布的參數（期望值）
    sigma_threshold = np.sqrt(lam)    # 收斂條件：標準差閾值
    normality_threshold = 0.05  # 常態性檢定 p-value 閾值

    # Process each sampling function
    for sampling_function, label in zip(sampling_functions, labels):
        sample_size = initial_sample_size
        converged = False

        while not converged:
            # Generate samples
            samples = sampling_function(lam, num_products * num_periods, sample_size)
            demand_samples = samples.reshape(sample_size, num_products, num_periods)

            # Compute sample statistics
            sample_mean = np.mean(demand_samples)
            sample_sizes[label].append(sample_size)
            means[label].append(sample_mean)

            sample_std = np.std(demand_samples)
            skewness =  skew(demand_samples)
            kurt = kurtosis(demand_samples)
            # 改用 Kolmogorov-Smirnov 檢驗            
            ks_statistic, ks_p_value = kstest(demand_samples.flatten(), 'norm', args=(sample_mean, sample_std))

            # Check convergence
            if abs(sample_mean - lam) < epsilon and abs(sample_std - sigma_threshold) < epsilon and ks_p_value < normality_threshold and sample_size > 400:
                converged = True
            else:
                sample_size += 50
                if sample_size > max_sample_size:
                    break

    # Plot combined mean convergence
    plt.figure(figsize=(10, 6))
    for label in labels:
        plt.plot(sample_sizes[label], means[label], label=f"{label} Sample Mean", marker="o")
    plt.axhline(y=lam, color="r", linestyle="--", label="Target Mean (λ)")
    plt.xlabel("Sample Size")
    plt.ylabel("Sample Mean")
    plt.title("Sample Mean Convergence: LHS vs RS")
    plt.legend()
    plt.grid()
    plt.show()



cumulative_fill_rate_mean, real_stock_mean, lhs_sample_size=fitness_lhs(order)
print(f"平均滿足率: {cumulative_fill_rate_mean}")
print(f"平均存貨量: {real_stock_mean}")


cumulative_fill_rate_mean, real_stock_mean, rs_sample_size=fitness_rs(order)
print(f"平均滿足率: {cumulative_fill_rate_mean}")       
print(f"平均存貨量: {real_stock_mean}")

print(f"lhs達收斂所需樣本數: {lhs_sample_size}") 
print(f"rs達收斂所需樣本數: {rs_sample_size}") 

lam=30
# Call the function to visualize combined mean convergence
visualize_combined_mean_convergence(
    order, lam, 
    [latin_hypercube_sampling, random_sampling], 
    ["LHS", "RS"]
)