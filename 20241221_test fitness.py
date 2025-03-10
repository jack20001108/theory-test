import numpy as np
import pandas as pd

#"零件"庫存與缺貨計算
def evaluator(demand, order, leadtime, begin_inventory, bom):
    #初始化矩陣
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
    #print(f"零件累積需求矩陣:{cumulative_demand}")

    #計算累積到貨
    arrival_order=np.roll(order, shift=leadtime, axis=1)
    arrival_order[:,:leadtime]=0 #在前leadtime期沒有補貨
    cumulative_arrival=np.cumsum(arrival_order, axis=1) #累積到貨
    #print(f"零件累積到達:{cumulative_arrival}")
 
    #計算每期庫存
    inventory[:,1:]=begin_inventory[:,None]+cumulative_arrival-cumulative_demand
    print("inv")
    print(inventory)
    #紀錄缺貨數量
    cummulate_shortage=np.where(inventory[:,1:]<0,-inventory[:,1:],0) #該期缺貨數量
    print("cum")
    print(cummulate_shortage)
    print("bom demand")
    print(bom@demand)
    shortage[:,0]=cummulate_shortage[:, 0]
    shortage[:,1:]=cummulate_shortage[:, 1:] - cummulate_shortage[:, :-1]
    #shortage=np.minimum(bom@demand, cummulate_shortage)
    print("short add ")
    print(shortage)
    bom_demand = bom[:, :, None] * demand  # 結果形狀: (N,M,T)
    # 計算反累積需求 (全部倒過來累加)
    shifted_bom_demand = np.roll(bom_demand, shift=-1, axis=1)  # 右移一格
    shifted_bom_demand[:, -1, :] = 0  # 將最後一列填充為 0

    reverse_cumulative_demand =np.cumsum(shifted_bom_demand[:, ::-1, :],axis=1)[:, ::-1, :]

    #reverse_cumulative_demand =np.cumsum(bom_demand[:, ::-1, :],axis=1)[:, ::-1, :]


    #reverse_to_demand=reverse_cumulative_demand.transpose(2,0,1)
    #print(f"reverse_to_demand 的形狀: {reverse_to_demand}")    
    #print(f"reverse_cumulative_demand 的形狀: {reverse_cumulative_demand}")
    reverse_inventory=inventory[:, 1:,None]
    reverse2_inventory=reverse_inventory.transpose(0,2,1)
 
    # 計算未分配庫存
    unclaim_inventory = (
        reverse2_inventory
        + reverse_cumulative_demand # 加入未分配需求
        
    ) 

    reshape_unclaim=unclaim_inventory.transpose(2,0,1)

    return inventory,shortage,reshape_unclaim
#    return inventory,shortage,unclaim_inventory



def fitness(order, leadtime, begin_inventory, bom, num_simulations=1000, demand_mean=20, demand_std=10, num_periods=5, num_products=3):
    """
    結合蒙地卡羅模擬，計算平均表現 (期望庫存狀態與缺貨表現)。
    
    :param order: 補貨計畫 (陣列，形狀為 (零件數量, 時期數))。
    :param leadtime: 補貨提前期 (整數)。
    :param begin_inventory: 初始庫存 (陣列，形狀為 (零件數量,))。
    :param bom: 需求 BOM 矩陣 (陣列，形狀為 (零件數量, 產品數量))。
    :param num_simulations: 模擬次數。
    :param demand_mean: 需求均值。
    :param demand_std: 需求標準差。
    :param num_periods: 時期數量。
    :param num_products: 產品種類數量。
    :return: 平均庫存狀態、平均缺貨數量。
    """
    # 隨機生成需求情境
    demands = np.random.normal(
        loc=demand_mean,
        scale=demand_std,
        size=(num_simulations, num_products, num_periods)
    ).astype(int)
    demands = np.maximum(demands, 0)  # 確保需求為非負數

    #建立零件與產品的三維矩陣
    #demand_reshaped = demands[None, :, :]  # 將 demand 形狀調整為 (1, N, T)
    # 計算每期每產品的 BOM 需求矩陣
    


    # 初始化結果
    total_inventory = np.zeros((begin_inventory.shape[0], num_periods + 1))
    total_shortage = np.zeros((begin_inventory.shape[0], num_periods))
    total_unclaim_inventory = np.zeros((num_periods, begin_inventory.shape[0], num_periods))

    #print("f/total unclaim")
    #print(total_unclaim_inventory)
    # 蒙地卡羅模擬
    for demand in demands:
        inventory, shortage, reshape_unclaim = evaluator(demand, order, leadtime, begin_inventory, bom)
        total_inventory += inventory
        total_shortage += shortage
        #total_unclaim_inventory += reshape_unclaim

    # 計算平均表現
    avg_inventory = total_inventory / num_simulations
    avg_shortage = total_shortage / num_simulations
    #avg_unclaim_inventory = total_unclaim_inventory / num_simulations

    return avg_inventory, avg_shortage






# 零件需求分配矩陣 (BOM)
bom = np.array([
    [1, 0, 1],  # 零件 1: 需要用於產品 1 和 3
    [2, 1, 0],  # 零件 2: 需要用於產品 1 和 2
    [0, 1, 1],  # 零件 3: 需要用於產品 2 和 3
    [1, 1, 0]   # 零件 4: 需要用於產品 1 和 2
])

# 產品需求矩陣
#demand = np.array([
#    [10, 15, 20, 25, 30],  # 產品 1 在每期的需求
#    [5, 10, 15, 10, 5],    # 產品 2 在每期的需求
#    [8, 12, 18, 22, 28]    # 產品 3 在每期的需求
#])

# 零件訂單到貨矩陣
order = np.array([
    [10, 20, 30, 10, 5],   # 零件 1 每期到貨數量
    [15, 25, 20, 15, 10],  # 零件 2 每期到貨數量
    [12, 18, 15, 12, 10],  # 零件 3 每期到貨數量
    [8, 10, 15, 20, 10]    # 零件 4 每期到貨數量
])

# 初始庫存
begin_inventory = np.array([50, 40, 30, 20])  # 零件 1, 2, 3, 4 的初始庫存

# 訂單前置時間
leadtime = 1

avg_inventory, avg_shortage = fitness(order, leadtime, begin_inventory, bom)

print("平均庫存狀態：")
print(avg_inventory)
print("\n平均缺貨數量：")
print(avg_shortage)




