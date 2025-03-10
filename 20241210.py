import numpy as np
import pandas as pd
#設定參數
np.random.seed(42) #隨機生成器 (呈現一致性)
num_sample=1000
num_period=30
hold_cost=1
order_cost=50
leadtime=3

# 使用週期訂貨 
# order_period=7
# order_quantity=100

#產生需求
##假設需求為常態分配
demand_mean=20
demand_std=5
demand_samples=np.random.normal(loc=demand_mean, scale=demand_std, size=(num_sample,num_period))
 # 確保需求非負
demand_samples = np.maximum(demand_samples, 0).astype(int) 

#產生訂貨
##假設訂貨為常態分配
order_mean=20
order_std=10
order=np.random.normal(loc=order_mean, scale=order_std, size=(num_sample,num_period))
order = np.maximum(order, 0).astype(int) 


#計算成本 
### 還要做的 1) 數學式要改成教授寫的樣子 2)加入lead time  3)變成兩層 
def cal_cost(demand,order,leadtime,hold_cost,order_cost):
    #初始值
    begin_inventory=100
    inventory=begin_inventory
    cumulate_demand=0
    cumulate_arrive=0
    total_hold_cost=0
    total_order_cost=0
    total_fill_demand=0
    total_shortage=0
    total_demand=np.sum(demand) 
    period=len(demand)


    #進行每期計算
    for period in range(len(demand)):

        print(f"begin inventory:{inventory}")         
 
        ##計算累積到貨
        if period>=leadtime:
            cumulate_arrive+=order[period-leadtime]      

        ## 計算累積需求
        cumulate_demand+=demand[period]       
        ##更新庫存
        inventory=begin_inventory+cumulate_arrive-cumulate_demand

        ##判斷滿足與缺貨數量
        if inventory>=0:
            fill_demand=demand[period]
            shortage=0
            total_hold_cost+=inventory*hold_cost ##計算持有成本
        elif demand[period]>abs(inventory):
            fill_demand=demand[period]+inventory
            shortage=demand[period]-fill_demand
            ##(補)計算缺貨影響
        else:
            fill_demand=0
            shortage=demand[period]
            ##(補)計算缺貨影響


        print("***********")
        print(f"period:{period}")
        print(f"cumulate_arrive:{cumulate_arrive}") 
        print(f"cumulate_onhand:{begin_inventory+cumulate_arrive}")         
        print(f"cumulate_demand:{cumulate_demand}") 
        print(f"inventory:{inventory}") 
        print(f"demand:{demand[period]}")
        print(f"fill_demand:{fill_demand}")
        print(f"shortage:{shortage}")
        print("***********")

 
        ## 計算總滿足/缺貨數量
        total_fill_demand+=fill_demand 
        total_shortage+=shortage    


        print(f"toatl_fill_demand:{total_fill_demand}")
        print(f"total_shortage:{total_shortage}")
        print("--------------------------------")

        ##計算訂購成本
        if order[period]>0:
            total_order_cost+=order_cost


    ## 計算總成本
    total_cost=total_hold_cost+total_order_cost
    #計算fill rate
    fill_rate=total_fill_demand/total_demand
    #完成
    print(f"total_demand:{total_demand}")
    return total_cost,fill_rate
        

## 測試
## 第一組數據
sample_demand=demand_samples[0]
sample_order=order[0]
#輸出生成數字
print("random demand")
print(sample_demand)
print("random order")
print(sample_order)


#輸出結果
total_cost,fill_rate=cal_cost(sample_demand,sample_order,leadtime,hold_cost,order_cost)

print(f"Total cost:{total_cost}")
print(f"fill rate:{fill_rate:.2%}")