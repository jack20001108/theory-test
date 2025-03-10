import numpy as np
from scipy.stats import norm
np.set_printoptions(suppress=True, precision=6)
import time


#"零件"庫存與缺貨計算
def evaluator(demand, order, leadtime, begin_inventory, bom):
    ##矩陣形狀
    num_product,num_period = demand.shape #產品數量,時間期數
    num_material=begin_inventory.shape[0] #零件數量
    ##庫存矩陣
    inventory = np.zeros((num_material, num_period+1)) # (M*T+1)
    inventory[:,0] = begin_inventory #初始庫存

    ##缺貨矩陣
    shortage=np.zeros((num_material, num_period)) #M*T

    #計算累積需求
    cumulative_demand = np.cumsum(bom@demand,axis=1)

    #計算累積到貨
    arrival_order=np.roll(order, shift=leadtime, axis=1)
    arrival_order[:,:leadtime]=0 #在前leadtime期沒有補貨
    cumulative_arrival=np.cumsum(arrival_order, axis=1) #累積到貨
    #print(f"零件累積到達:{cumulative_arrival}")
 
    #計算每期庫存
    inventory[:,1:]=begin_inventory[:,None]+cumulative_arrival-cumulative_demand
        #print("inv")
        #print(inventory)
    #紀錄缺貨數量
    cummulate_shortage=np.where(inventory[:,1:]<0,-inventory[:,1:],0) #該期缺貨數量
        #print("cum")
        #print(cummulate_shortage)
        #print("bom demand")
        #print(bom@demand)
    shortage[:,0]=cummulate_shortage[:, 0]
    shortage[:,1:]=cummulate_shortage[:, 1:] - cummulate_shortage[:, :-1]
    #shortage=np.minimum(bom@demand, cummulate_shortage)
        #print("short add ")
        #print(shortage)
    bom_demand = bom[:, :, None] * demand  # 結果形狀: (N,M,T)
    # 計算反累積需求 (全部倒過來累加)
    shifted_bom_demand = np.roll(bom_demand, shift=-1, axis=1)  # 右移一格
    shifted_bom_demand[:, -1, :] = 0  # 將最後一列填充為 0

    reverse_cumulative_demand =np.cumsum(shifted_bom_demand[:, ::-1, :],axis=1)[:, ::-1, :]

    reverse_inventory=inventory[:, 1:,None]
    reverse2_inventory=reverse_inventory.transpose(0,2,1)
 
    # 計算未分配庫存
    unclaim_inventory = (
        reverse2_inventory
        + reverse_cumulative_demand # 加入未分配需求
        ) 

    reshape_unclaim=unclaim_inventory.transpose(2,0,1)

    #每個產品，所需零件在各期間是否缺貨 (矩陣代表產品N)
    reshape_to_product=unclaim_inventory.transpose(1,0,2)
        #print("reshape_to_product：",reshape_to_product) 



        #計算零件滿足率
    aligned_bom_demand = bom_demand.transpose(2,0,1)
    shortage_com = np.where(
        reshape_unclaim < 0,
        np.minimum(-reshape_unclaim, aligned_bom_demand),
        0
    )

    # 避免分母為零
    safe_aligned_bom_demand = np.where(aligned_bom_demand == 0, 1e-10, aligned_bom_demand)

        #print("shortage_com")
        #print(shortage_com)
    
    shortage_com_tr=shortage_com.transpose(2,1,0)
        #print("shortage_com_tr")
        #print(shortage_com_tr)

    return inventory,shortage,reshape_unclaim,shortage_com,safe_aligned_bom_demand,shortage_com_tr



def latin_hypercube_sampling(dimensions, samples):

    result = np.zeros((samples, dimensions))
    for i in range(dimensions):
        cut = np.linspace(0, 1, samples + 1)  # 分層切割
        uniform_samples = np.random.uniform(cut[:-1], cut[1:], size=samples) #每個子區間的均勻隨機樣本
        np.random.shuffle(uniform_samples) # 打散隨機樣本
        result[:, i] = uniform_samples # 填入結果
    return result

def fitness(
    
    num_products, initial_samples=1000,
    sample_increment=50, max_iterations=10, target_precision=0.01,num_periods=5):
    

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

        #print("demand_samples")
        #print(demand_samples)
        # 確保形狀正確
        demand_samples = demand_samples.reshape(total_samples, num_products, num_periods)
        #print("demand_samples reshape")
        #print(demand_samples)
        # 記錄當前樣本的填充率
        fill_rates = []  # 當前樣本的填充率
        for demand in demand_samples:
            # 每組樣本分別傳入 fitness，demand 的形狀為 (num_products, num_periods)
            # 呼叫 evaluator 函數

            ininventory,shortage,reshape_unclaim,shortage_com,safe_aligned_bom_demand,shortage_com_tr = evaluator(demand, order, leadtime, begin_inventory, bom)


            #shortage_com_tr
            # 計算 fill rate
            com_fill_rate = 1 - (shortage_com / safe_aligned_bom_demand)
            com_fill_rate_tr = com_fill_rate.transpose(2,1,0)
            pro_fill_rate=np.prod(com_fill_rate_tr,axis=1) #產品填充率
            fill_rates.append(pro_fill_rate)
            

            # 動態計算累積平均填充率
            cumulative_fill_rate.append(pro_fill_rate)
            #print("cumulative_fill_rate")
            #print(cumulative_fill_rate)
            cumulative_mean = np.mean(cumulative_fill_rate,axis=0)




        # 如果已執行了至少 1000 組樣本，開始進行收斂檢查
        if len(cumulative_fill_rate) >= 1000:
            # 即時檢查收斂性
            if previous_cumulative_mean is not None:
                relative_error = np.abs(cumulative_mean - previous_cumulative_mean) 
                #print(f"第 {iteration + 1} 次迭代: 累積平均填充率 = {cumulative_mean}")
                #print("相對誤差 = ", relative_error)


                if np.all(relative_error < target_precision):
                    #print(f"第 {iteration + 1} 次迭代: 收斂，終止模擬")
                    return cumulative_fill_rate, cumulative_mean, len(cumulative_fill_rate)


                
            # 更新前一次累積平均值
            previous_cumulative_mean = cumulative_mean
        #print(f"共使用樣本數：{total_samples}")
        # 增加樣本數
        total_samples += sample_increment
    




    return cumulative_fill_rate, cumulative_mean, len(cumulative_fill_rate)


# 模擬參數
num_products = 3  # 產品數量
mean_demand = np.random.randint(10, 50, size=num_products)  # 每個產品的需求均值
std_dev_demand = np.random.randint(5, 15, size=num_products)  # 每個產品的需求標準差

# 示例參數
order = np.array([
    [10, 20, 30, 10, 5],
    [15, 25, 20, 15, 10],
    [12, 18, 15, 12, 10],
    [8, 10, 15, 20, 10]
])
begin_inventory = np.array([100, 90, 80, 70])
leadtime = 1
bom = np.array([
    [1, 0, 1],
    [2, 1, 0],
    [0, 1, 1],
    [1, 1, 0]
])

# 執行模擬
start_time = time.time()
samples, final_shortage_mean, total_samples = fitness(
    num_products, initial_samples=1000,
    sample_increment=50, max_iterations=10, target_precision=0.01,num_periods=5)

# 記錄結束時間
end_time = time.time()
# 計算執行時間
execution_time = end_time - start_time
print(f"模擬完成，共使用樣本數：{total_samples}")
print(f"最終平均滿足率：{final_shortage_mean}")
print(f"執行時間：{execution_time:.4f} 秒")



