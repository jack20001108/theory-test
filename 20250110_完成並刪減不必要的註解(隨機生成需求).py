import numpy as np
from scipy.stats import norm
np.set_printoptions(suppress=True, precision=6)
import time



def evaluator(demand, order, leadtime, begin_inventory, bom):  # 輸入一組需求與訂購資料 輸出缺貨矩陣(缺貨時間,缺貨數量) 
    ##建立矩陣
    num_product,num_period = demand.shape #產品數量,時間期數
    num_material=begin_inventory.shape[0] #零件數量
    bom_demand = bom[:, :, None] * demand  #零件需求展開(N,M,T)
    inventory = np.zeros((num_material, num_period+1)) #庫存(M*T+1)
    inventory[:,0] = begin_inventory #初始庫存
    shortage=np.zeros((num_material, num_period)) #缺貨(M*T)
    #計算庫存要用到的數據
    material_demand = bom_demand.transpose(2,0,1) #每個產品的需求(T,N,M)
    cumulative_demand = np.cumsum(bom@demand,axis=1) #累積需求
    arrival_order=np.roll(order, shift=leadtime, axis=1) #在leadtime期補貨
    arrival_order[:,:leadtime]=0 #在前leadtime期沒有補貨
    cumulative_arrival=np.cumsum(arrival_order, axis=1) #累積到貨
    #計算每期庫存
    inventory[:,1:]=begin_inventory[:,None]+cumulative_arrival-cumulative_demand
    #紀錄缺貨數量
    cummulate_shortage=np.where(inventory[:,1:]<0,-inventory[:,1:],0) #該期缺貨數量
    shortage[:,0]=cummulate_shortage[:, 0]
    shortage[:,1:]=cummulate_shortage[:, 1:] - cummulate_shortage[:, :-1]
    # 計算反累積需求 (全部倒過來累加)
    shifted_bom_demand = np.roll(bom_demand, shift=-1, axis=1)  #右移一格
    shifted_bom_demand[:, -1, :] = 0  #將最後一列補0
    reverse_cumulative_demand =np.cumsum(shifted_bom_demand[:, ::-1, :],axis=1)[:, ::-1, :] #反累積需求(時間軸轉回來)
    reverse_inventory=inventory[:, 1:,None] 
    reverse2_inventory=reverse_inventory.transpose(0,2,1) 
    #計算未分配零件數量
    unclaim_inventory = ( reverse2_inventory + reverse_cumulative_demand) #加入未分配需求 
    material_unclaim=unclaim_inventory.transpose(2,0,1)
    #產品缺貨矩陣
    product_shortage=unclaim_inventory.transpose(1,0,2) #每個產品，所需零件在各期間是否缺貨 

    return material_demand,inventory,shortage,material_unclaim,product_shortage



def latin_hypercube_sampling(dimensions, samples): #輸入欲生成的維度、樣本數 輸出LHS矩陣
    result = np.zeros((samples, dimensions))
    for i in range(dimensions):
        cut = np.linspace(0, 1, samples + 1)  #切割成樣本數量的空間
        uniform_samples = np.random.uniform(cut[:-1], cut[1:], size=samples) #每個區間產生均勻隨機樣本
        np.random.shuffle(uniform_samples) #順序隨機打散
        result[:, i] = uniform_samples 
    return result

def fitness(num_products=3,num_periods=5):
    #隨機樣本生成設定
    initial_samples=1000
    sample_increment=50 
    max_iterations=10
    min_demand, max_demand = 10, 50
    target_precision=0.001
    #儲存矩陣
    cumulative_fill_rate = []  #累積產品滿足率
    previous_cumulative_mean = None  # 上一個迭代的累積平均
    ##(生成初始樣本>計算平均>收斂檢查>迭代至收斂)
    #生成初始樣本
    lhs_samples = latin_hypercube_sampling(num_products * num_periods, initial_samples)
    demand_samples = np.floor(lhs_samples * (max_demand - min_demand + 1) + min_demand).astype(int)
    demand_samples = demand_samples.reshape(initial_samples, num_products, num_periods)
    #計算初始樣本的產品滿足率
    for demand in demand_samples:
        material_demand,inventory,shortage,material_unclaim,product_shortage = evaluator(demand, order, leadtime, begin_inventory, bom)
        shortage_com = np.where(material_unclaim < 0, np.minimum(-material_unclaim, material_demand),0)
        safe_material_demand = np.where(material_demand == 0, 1e-10, material_demand) #避免分母為零
        shortage_com_tr=shortage_com.transpose(2,1,0)
        com_fill_rate = 1 - (shortage_com / safe_material_demand)
        com_fill_rate_tr = com_fill_rate.transpose(2,1,0)
        pro_fill_rate = np.prod(com_fill_rate_tr, axis=1) #產品滿足率
        cumulative_fill_rate.append(pro_fill_rate)
    #迭代生成新增樣本
    for iteration in range(max_iterations):
        lhs_samples = latin_hypercube_sampling(num_products * num_periods, sample_increment)
        demand_samples = np.floor(lhs_samples * (max_demand - min_demand + 1) + min_demand).astype(int)
        demand_samples = demand_samples.reshape(sample_increment, num_products, num_periods)
        #計算新增樣本的產品滿足率
        for demand in demand_samples:
            material_demand,inventory,shortage,material_unclaim,product_shortage= evaluator(demand, order, leadtime, begin_inventory, bom)
            shortage_com = np.where(material_unclaim < 0,np.minimum(-material_unclaim, material_demand),0)
            safe_material_demand = np.where(material_demand == 0, 1e-10, material_demand) #避免分母為零
            shortage_com_tr=shortage_com.transpose(2,1,0)
            com_fill_rate = 1 - (shortage_com / safe_material_demand)
            com_fill_rate_tr = com_fill_rate.transpose(2,1,0)
            pro_fill_rate = np.prod(com_fill_rate_tr, axis=1)  # 產品滿足率
            cumulative_fill_rate.append(pro_fill_rate)
        #計算累積平均產品滿足率
        cumulative_fill_rate_array = np.array(cumulative_fill_rate)
        cumulative_mean = np.mean(cumulative_fill_rate_array, axis=0)

        #收斂檢查
        if len(cumulative_fill_rate) >= 1000:
            if previous_cumulative_mean is not None:
                relative_error = np.abs(cumulative_mean - previous_cumulative_mean)
                if np.all(relative_error < target_precision):
                    return cumulative_fill_rate_array, cumulative_mean, len(cumulative_fill_rate)
            previous_cumulative_mean = cumulative_mean

    return cumulative_fill_rate_array, cumulative_mean, len(cumulative_fill_rate)



# 設定參數
num_products = 3  
num_periods = 5  
leadtime = 1
begin_inventory = np.array([100, 90, 80, 70])
order = np.array([
    [10, 20, 30, 10, 5],
    [15, 25, 20, 15, 10],
    [12, 18, 15, 12, 10],
    [8, 10, 15, 20, 10]
])

bom = np.array([
    [1, 0, 1],
    [2, 1, 0],
    [0, 1, 1],
    [1, 1, 0]
])


start_time = time.time()
samples, final_shortage_mean, total_samples= fitness(num_products,num_periods)
end_time = time.time()
execution_time = end_time - start_time  # 計算執行時間


print(f"模擬完成，共使用樣本數：{total_samples}")
print(f"最終平均滿足率：{final_shortage_mean}")
print(f"執行時間：{execution_time:.4f} 秒")