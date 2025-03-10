import numpy as np

# "零件"庫存與缺貨計算
def evaluator(demand, order, leadtime, begin_inventory, bom):
    #矩陣形狀 (define metrix shape)
    num_simulations, num_products, num_periods = demand.shape #三維(模擬次數K,產品數量N,期數T)
    num_material = begin_inventory.shape[0] #零件數量 

    #初始庫存矩陣
    inventory = np.zeros((num_simulations, num_material, num_periods + 1))  # K*M*(T+1) 
    inventory[:, :, 0] = begin_inventory #初始庫存
    print("demand：", demand)    
    #計算零件需求
    component_demand = bom[:, :, None] * demand[:, None, :, :] # K*M*N*T
    print("component_demand：", component_demand)
    #計算累積需求
    cumulate_demand = np.cumsum(component_demand.sum(axis=2), axis=2)  # K*M*T
    print("cumulate_demand：", cumulate_demand)

    #計算累積到貨
    arrival_order = np.roll(order, shift=leadtime, axis=1)  #滾動補貨前置期
    arrival_order[:, :leadtime] = 0  #前 leadtime 期無到貨
    arrival_order = np.expand_dims(arrival_order, axis=0).repeat(num_simulations, axis=0)  # 擴展至K模擬情境
    cumulate_arrival = np.cumsum(arrival_order, axis=2)  # K*M*T
    print("cumulate_arrival：", cumulate_arrival)
    #計算每期庫存
    inventory[:, :, 1:] = begin_inventory[None, :, None] + cumulate_arrival - cumulate_demand

    #計算缺貨
    shortage = np.where(inventory[:, :, 1:] < 0, -inventory[:, :, 1:], 0)  #該期缺貨數量

    #計算未分配需求
    shift_bom_demand=np.roll(component_demand,shift=-1,axis=1) 
    shift_bom_demand[:,-1,:,:]=0

    reverse_cumulative_demand = np.cumsum(component_demand[:, ::-1, :, :], axis=1)[:, ::-1, :, :]  #反累積需求
    print("reverse_cumulative_demand：",reverse_cumulative_demand)
    #刪除第0個矩陣
    modified_cumulative_demand = reverse_cumulative_demand[:, 1:, :, :]

    #補上零矩陣
    zero_matrix = np.zeros((num_simulations, 1, num_products, num_periods))
    adjusted_reverse_cumulative_demand = np.concatenate([modified_cumulative_demand, zero_matrix],axis=1 )
    print("adjusted_reverse_cumulative_demand：",adjusted_reverse_cumulative_demand)
    reverse_inventory=inventory[:, :, 1:,None]
    reverse2_inventory=reverse_inventory.transpose(0,1,3,2)
    print("reverse2_inventory：",reverse2_inventory)

    unclaim_inventory = (
        reverse2_inventory
        + adjusted_reverse_cumulative_demand  #未分配庫存
    )
    print("unclaim_inventory：",unclaim_inventory)
    #每時間點，零件分配角度看是否缺貨(矩陣代表時間T=1,2,...,t)
    reshape_unclaim=unclaim_inventory.transpose(0,3,1,2)
    print("reshape_unclaim：",reshape_unclaim)
    #每個產品，所需零件在各期間是否缺貨 (矩陣代表產品N)
    reshape_to_product=unclaim_inventory.transpose(0,2,1,3)
    print("reshape_to_product：",reshape_to_product)    


    #計算零件滿足率
    aligned_bom_demand = component_demand.transpose(0,3,1,2)
    shortage_com = np.where(
        reshape_unclaim < 0,
        np.minimum(-reshape_unclaim, aligned_bom_demand),
        0
    )

    # 避免分母為零
    safe_aligned_bom_demand = np.where(aligned_bom_demand == 0, 1e-10, aligned_bom_demand)

    # 計算 fill rate
    com_fill_rate = 1 - (shortage_com / safe_aligned_bom_demand)
    print ("inventory")
    print(inventory)
    print ("shortage")
    print(shortage)
    print ("unclaim_inventory")
    print(unclaim_inventory)
    print ("com_fill_rate")
    print(com_fill_rate)    

    return inventory, shortage, unclaim_inventory, com_fill_rate



def fitness(order, leadtime, begin_inventory, bom, num_simulations=2, demand_mean=15, demand_std=5, num_periods=5, num_products=3):

    # 隨機生成需求情境 (模擬所有情境)
    demands = np.random.normal(
        loc=demand_mean,
        scale=demand_std,
        size=(num_simulations, num_products, num_periods)
    ).astype(int)
    demands = np.maximum(demands, 0)  # 確保需求為非負數

    # 呼叫 evaluator 函數
    inventory, shortage, unclaim_inventory, com_fill_rate = evaluator(demands, order, leadtime, begin_inventory, bom)

    # 平均庫存與缺貨
    avg_inventory = inventory.mean(axis=0)  # 平均庫存
    avg_shortage = shortage.mean(axis=0)  # 平均缺貨

    # 計算產品層級的平均滿足率
    product_fill_rate_by_period = np.prod(com_fill_rate, axis=2).mean(axis=0)  # 每期平均滿足率
    average_product_fill_rate = np.mean(product_fill_rate_by_period, axis=0)  # 總平均滿足率
    all_fill_rate=np.mean(average_product_fill_rate,axis=0)    
    #print(inventory.shape)
    return avg_inventory, avg_shortage, com_fill_rate, product_fill_rate_by_period, average_product_fill_rate
    #return all_fill_rate


# 測試參數
bom = np.array([
    [1, 0, 1],
    [2, 1, 0],
    [0, 1, 1],
    [1, 1, 0]
])
order = np.array([
    [10, 20, 30, 10, 5],
    [15, 25, 20, 15, 10],
    [12, 18, 15, 12, 10],
    [8, 10, 15, 20, 10]
])
begin_inventory = np.array([50, 40, 30, 20])
leadtime = 1

# 執行 fitness
avg_inventory, avg_shortage, com_fill_rate, product_fill_rate_by_period, average_product_fill_rate = fitness(
    order, leadtime, begin_inventory, bom
)

print("平均庫存狀態：", avg_inventory)
print("平均缺貨數量：", avg_shortage)
print("\n零件滿足率矩陣:")
print(com_fill_rate)
print("每期每產品的平均滿足率：", product_fill_rate_by_period)
print("每產品的平均滿足率：", average_product_fill_rate)
print(com_fill_rate.shape)