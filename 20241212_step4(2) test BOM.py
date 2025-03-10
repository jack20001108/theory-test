import numpy as np
import pandas as pd

# 非註解文字的程式分兩種
#### (四個#) : 檢查的程式
##   (兩個#) : 可替換程式(想到先放著//不一定正確)

#設定參數
np.random.seed(42) #隨機生成器 (呈現一致性)
num_sample=1000
num_period=30
hold_cost=1
order_cost=50
leadtime=3
begin_inventory=100
confidence_level=0.95

#產生需求
##假設需求為常態分配(可假設其他分配)
###隨機生成三種不同最終產品的需求
demand_mean=[20,30,40]
demand_std=[5,6,7]
demand_samples = [
    np.maximum(np.random.normal(loc=mean, scale=std, size=(num_sample, num_period)), 0).astype(int)
    for mean, std in zip(demand_mean, demand_std)
]

#將最終產品的需求轉換成矩陣
product_demand=np.array(demand_samples)

#定義BOM
bom_matrix=np.array([
    [2,1,0], #product1 
    [1,2,1], #product2
    [0,1,2]  #product3
])

components=["comX","comY","comZ"]
finish_product=["pro1","pro2","pro3"]

bom_df=pd.DataFrame(bom_matrix,index=finish_product,columns=components)



#利用BOM表計算需求
def cal_component_demand(product_demand,bom_matrix):
    #最終產品需求形狀轉換
    reshape_product_demand=product_demand.reshape(3,-1)
    #計算零件需求
    component_demand_matrix=bom_matrix.T @ reshape_product_demand
    #變回(3,num_sample,num,period形狀) 
    component_demand_matrix=component_demand_matrix.reshape(3,num_sample,num_period)

    return component_demand_matrix

component_demand=cal_component_demand(product_demand,bom_matrix)

#轉換成dataframe呈現
for i,component in enumerate(components):
    print(f"{component}demand number:{component_demand}")


## 測試
## 第一組數據
def test_first_sample():
    #抓第一組的資料
    sample_product_demand=product_demand[:,0,:]

    product_demand_dfs={}
    #輸出第一組產品的需求
    for i ,product in enumerate(finish_product):
        df_p=pd.DataFrame(
            sample_product_demand[i].reshape(1,-1),
            columns=[f"period{i+1}"for i in range(num_period)]
        )
        product_demand_dfs[product]=df_p
    #輸出每個零件的需求
    for product,df_p in product_demand_dfs.items():
        print(f"產品{product}的需求在第一組樣本為:")
        print(df_p)
        





    # 計算第一組樣本的零件需求
    reshape_product_demand = sample_product_demand.reshape(3, -1)  # 重塑需求形狀為 (3 x num_period)
    sample_component_demand = bom_matrix.T @ reshape_product_demand  # 矩陣乘法 (3 x num_period)

    print(f"---------------------------------------------------")

    #將零件的需求轉換為DataFrame 進行檢查
    component_demand_dfs={}

    for i ,component in enumerate(components):
        df_c=pd.DataFrame(
            sample_component_demand[i].reshape(1,-1),
            columns=[f"period{i+1}"for i in range(num_period)]
        )
        component_demand_dfs[component]=df_c

    #輸出每個零件的需求
    for component,df_c in component_demand_dfs.items():
        print(f"零件{component}的需求在第一組樣本為:")
        print(df_c)

## 輸出bom表資料
print("BOM matrix")
print(bom_df)

# 呼叫測試第一組
test_first_sample()

