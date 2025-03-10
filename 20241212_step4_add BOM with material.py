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
    #變回(3,num_sample,num,period形狀) #### 為何不直接用矩陣就好???????????????????
    component_demand_matrix=component_demand_matrix.reshape(3,num_sample,num_period)

    return component_demand_matrix

component_demand=cal_component_demand(product_demand,bom_matrix)

#轉換成dataframe呈現
for i,component in enumerate(components):
    total_demand=component_demand[i].sum()
    print(f"{component}demand number:{total_demand}")




#產生訂貨
##給定一組解 (此為零件訂貨數量)
order=[
    [ 0,3,10,30,35,0,10,5,29,0,70,2,50,8,2,0,80,6,70,32,14,50,20,40,20,30,33,0,0,20,4],
    [ 0,4,20,40,60,80,31,15,99,34,6,50,59,0,0,100,50,66,70,38,10,5,60,60,0,20,3,10,10,45,0],
    [ 0,5,34,3,68,8,19,54,85,80,70,9,98,86,70,30,68,0,70,39,0,78,28,44,12,39,37,15,20,29,8]
]

## 其他方法 (好像沒必要了??)
### 方法一: 假設訂貨為常態分配
#### order_mean=20
#### order_std=10
#### order=np.random.normal(loc=order_mean, scale=order_std, size=(num_sample,num_period))
#### order = np.maximum(order, 0).astype(int) 
### 方法二: 使用週期訂貨 
#### order_period=7
#### order_quantity=100

#計算成本 
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

        ####print("***********")
        ####print(f"period:{period}")
        ####print(f"cumulate_arrive:{cumulate_arrive}") 
        ####print(f"cumulate_onhand(beg+arr):{begin_inventory+cumulate_arrive}")         
        ####print(f"cumulate_demand:{cumulate_demand}") 
        ####print(f"end_inventory:{inventory}") 
        ####print(f"demand:{demand[period]}")
        ####print(f"fill_demand:{fill_demand}")
        ####print(f"shortage:{shortage}")
        ####print(f"individal_fill_rate:{indiv_fill_rate:.2%}")
        ##print(f"shortage2:{shortage_2}")
        ##print(f"fill_demand_2:{fill_demand_2}") 
        ####print("***********")

 
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
        


#蒙地卡羅模擬
result=[]
for product_id in range(3):
    print(f"product:{product_id+1}")
    product_costs=[]
    product_fill_rates=[]
    product_shortages=[]    #定義損失為缺貨數量
    product_shortages_period=[]     #定義損失為缺貨期數

    for i in range(num_sample):
        sample_demand=demand_samples[product_id][i]
        print("--------------------------------")    
        print(f"sample:{i+1}")
        total_cost,prod_fill_rate,total_shortage_end,cumulate_shortage_time=cal_cost(sample_demand, order[product_id], leadtime, hold_cost, order_cost)
        print(f"Total cost:{total_cost}")
        print(f"fill rate:{prod_fill_rate:.2%}")
        print(f"Total shortage end:{total_shortage_end}")
        print(f"Total shortage time:{cumulate_shortage_time}")   
        #蒐集結果存在列表
        product_costs.append(total_cost)
        product_fill_rates.append(prod_fill_rate)
        product_shortages.append(total_shortage_end)
        product_shortages_period.append(cumulate_shortage_time)
    expected_cost=np.mean(product_costs)
    average_fill_rate=np.mean(product_fill_rates)
    #缺貨數量的VaR/CVaR
    risk_var_sn=np.percentile(product_shortages,100*confidence_level)
    risk_cvar_sn=np.mean([s for s in product_shortages if s>risk_var_sn])
    #缺貨期數的VaR/CVaR
    risk_var_sp=np.percentile(product_shortages_period,100*confidence_level)
    risk_cvar_sp=np.mean([s for s in product_shortages_period if s>risk_var_sp])
    #持有成本
    print("--------------------------------")
    print(f"VaR(shortage number):{risk_var_sn}")
    print(f"CVaR(shortage number):{risk_cvar_sn}")
    print(f"VaR(shortage period):{risk_var_sp}")
    print(f"CVaR(shortage period):{risk_cvar_sp}")
    print(f"expected cost:{expected_cost}")
    print(f"average fill rate:{average_fill_rate:.2%}")

    result.append({
        "Product": f"Product {product_id + 1}",
        "Expected Cost": expected_cost,
        "Average Fill Rate": f"{average_fill_rate:.2%}",
        "VaR (Shortage Number)": risk_var_sn,
        "CVaR (Shortage Number)": risk_cvar_sn,
        "VaR (Shortage Period)": risk_var_sp,
        "CVaR (Shortage Period)": risk_cvar_sp
    })

result_df=pd.DataFrame(result)
print("-------------result---table--------------------")
print(result_df.to_string())

## 輸出bom表資料
print("BOM matrix")
print(bom_df)