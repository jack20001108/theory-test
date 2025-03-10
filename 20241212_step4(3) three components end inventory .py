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


#產生訂貨
##給定一組解 (此為零件訂貨數量)
order = [
    [230, 45, 90, 57, 66, 100, 21, 9, 81, 0, 138, 3, 129, 14, 5, 0, 209, 9, 159, 63, 33, 81, 55, 114, 50, 47, 60, 0, 0, 35, 10],
    [800, 80, 52, 74, 167, 159, 48, 24, 201, 84, 15, 125, 89, 0, 0, 236, 146, 135, 159, 78, 27, 9, 131, 159, 0, 42, 5, 17, 21, 126, 0],
    [400, 11, 80, 8, 164, 14, 47, 134, 203, 194, 197, 26, 192, 243, 129, 52, 153, 0, 158, 114, 0, 194, 50, 122, 20, 116, 57, 36, 47, 66, 12]
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





#計算期末庫存,成本,fill rate
def cal_cost(demand,order,leadtime,hold_cost,order_cost):
    #初始值
    inventory=begin_inventory
    cumulate_demand=0
    cumulate_arrive=0
    total_hold_cost=0
    total_order_cost=0
    total_fill_demand=0
    total_shortage=0
    total_demand=np.sum(demand) 
    period=len(demand)
    cumulate_shortage_time=0

    #進行每期計算
    for period in range(len(demand)):


        ####print(f"begin inventory:{inventory}")         
 
        #計算累積到貨
        if period>=leadtime:
            cumulate_arrive+=order[period-leadtime]      

        #計算累積需求
        cumulate_demand+=demand[period]       
        #更新庫存
        inventory=begin_inventory+cumulate_arrive-cumulate_demand

        #判斷滿足與缺貨數量
        if inventory>=0:
            fill_demand=demand[period]
            shortage=0
            total_hold_cost+=inventory*hold_cost ##計算持有成本
        elif demand[period]>abs(inventory):
            fill_demand=demand[period]+inventory
            shortage=demand[period]-fill_demand
            cumulate_shortage_time+=1  ##計數器(計算缺貨期數)
            ##(補)計算缺貨影響
        else:
            fill_demand=0
            shortage=demand[period]
            cumulate_shortage_time+=1 ##計數器(計算缺貨期數)
            ##(補)計算缺貨影響
        
        #我的數學式(前期庫存為負時會出現錯誤????)
        ##shortage_2=max(-1*inventory,0)
        ##fill_demand_2=demand[period]-shortage

        #計算individual fill rate
        ##indiv_fill_rate=1-(shortage/demand[period])

        print("***********")
        print(f"period:{period}")
        print(f"cumulate_arrive:{cumulate_arrive}") 
        print(f"cumulate_onhand(beg+arr):{begin_inventory+cumulate_arrive}")         
        print(f"cumulate_demand:{cumulate_demand}") 
        print(f"end_inventory:{inventory}") 
        print(f"demand:{demand[period]}")
        print(f"fill_demand:{fill_demand}")
        print(f"shortage:{shortage}")
        ####print(f"individal_fill_rate:{indiv_fill_rate:.2%}")
        ##print(f"shortage2:{shortage_2}")
        ##print(f"fill_demand_2:{fill_demand_2}") 
        print("***********")

 
        #計算總滿足/缺貨數量
        total_fill_demand+=fill_demand 
        total_shortage+=shortage    
        ####print(f"toatl_fill_demand:{total_fill_demand}")
        ####print(f"total_shortage:{total_shortage}")
        ####print("--------------------------------")

        #計算訂購成本
        if period<len(order) and order[period]>0:
            total_order_cost+=order_cost

    total_shortage_end=total_shortage
    #計算總成本
    total_cost=total_hold_cost+total_order_cost
    #計算product fill rate
    prod_fill_rate=total_fill_demand/total_demand
    print(f"toatl_fill_demand:{total_fill_demand}")
    print(f"total_demand:{total_demand}")
    return total_cost,prod_fill_rate,total_shortage_end,cumulate_shortage_time



## 測試
## 第一組數據
# 測試第一組隨機需求
result = []

for component_id in range(3):
    print(f"Testing Product {component_id + 1}")

    # 抓取第一組隨機需求
    sample_demand_to_component = component_demand[component_id][0]
    print("--------------------------------")
    print(f"Sample 1 Demand for component {component_id + 1}:")
    print(sample_demand_to_component)

    # 計算成本、fill rate 和缺貨資訊
    total_cost, prod_fill_rate, total_shortage_end, cumulate_shortage_time = cal_cost(
        sample_demand_to_component, order[component_id], leadtime, hold_cost, order_cost
    )

    # 輸出結果
####    print(f"Total Cost: {total_cost}")
####    print(f"Fill Rate: {prod_fill_rate:.2%}")
####    print(f"Total Shortage End: {total_shortage_end}")
####    print(f"Total Shortage Time: {cumulate_shortage_time}")

    # 儲存結果
    result.append({
        "component": f"component {component_id + 1}",
        "Total Cost": total_cost,
        "Fill Rate": f"{prod_fill_rate:.2%}",
        "Total cummulate shortage": total_shortage_end,
        "Total Shortage Time": cumulate_shortage_time
    })

# 將結果轉為 DataFrame 並輸出
result_df = pd.DataFrame(result)
print("-------------Result Table--------------------")
print(result_df.to_string())

## 輸出bom表資料
print("BOM matrix")
print(bom_df)