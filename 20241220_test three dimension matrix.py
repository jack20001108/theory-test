import numpy as np

# 零件需求分配矩陣 (BOM)
bom = np.array([
    [1, 0, 1],  # 零件 1: 需要用於產品 1 和 3
    [2, 1, 0],  # 零件 2: 需要用於產品 1 和 2
    [0, 1, 1],  # 零件 3: 需要用於產品 2 和 3
    [1, 1, 0]   # 零件 4: 需要用於產品 1 和 2
])

# 產品需求矩陣
demand = np.array([
    [10, 15, 20, 25, 30],  # 產品 1 在每期的需求
    [5, 10, 15, 10, 5],    # 產品 2 在每期的需求
    [8, 12, 18, 22, 28]    # 產品 3 在每期的需求
])

# 計算每期每產品的 BOM 需求矩陣
# 調整形狀以進行正確的廣播
demand_reshaped = demand[None, :, :]  # 將 demand 形狀調整為 (1, N, T)

# 計算每期每產品的 BOM 需求矩陣
bom_demand = bom[:, :, None] * demand_reshaped  # 結果形狀: (N,M,T)
print(bom_demand)
print("零件與產品的需求矩陣 (4x3):")
print(bom_demand[0,:,:])
print(bom_demand[:,0,:])
print(bom_demand[:,:,0])

shifted_bom_demand = np.roll(bom_demand, shift=-1, axis=1)  # 右移一格
shifted_bom_demand[:, -1, :] = 0  # 將最後一列填充為 0

reverse_cumulative_demand =np.cumsum(shifted_bom_demand[:, ::-1, :],axis=1)[:, ::-1, :]
reshape_rev=reverse_cumulative_demand.transpose(2,0,1)
print(reverse_cumulative_demand)
print(reshape_rev)
print(reverse_cumulative_demand[:,:,0])