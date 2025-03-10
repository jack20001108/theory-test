import numpy as np

# 初始化參數
initial_inventory = 50
demand_mean = 10
demand_std = 3
lead_time = 2
target_inventory = 100
reorder_threshold = 40
check_frequency = 5
simulation_days = 30
num_simulations = 1000

def simulate_inventory(strategy="check"):
    fill_rate_results = []
    total_costs = []
    for _ in range(num_simulations):
        inventory = initial_inventory
        backorders = 0
        total_cost = 0
        fulfilled_demand = 0
        total_demand = 0
        orders = []

        for day in range(simulation_days):
            # 每日需求
            daily_demand = max(0, np.random.normal(demand_mean, demand_std))
            total_demand += daily_demand

            # 減去需求
            if inventory >= daily_demand:
                inventory -= daily_demand
                fulfilled_demand += daily_demand
            else:
                backorders += daily_demand - inventory
                fulfilled_demand += inventory
                inventory = 0

            # 補貨到達
            if orders and day == orders[0][0]:
                inventory += orders.pop(0)[1]

            # 檢查補貨
            if strategy == "check" and day % check_frequency == 0:
                if inventory < reorder_threshold:
                    order_quantity = max(0, target_inventory - inventory)
                    orders.append((day + lead_time, order_quantity))

        # 計算 fill rate 和成本
        fill_rate = fulfilled_demand / total_demand
        total_cost = backorders * 10 + sum(o[1] * 2 for o in orders)  # 缺貨成本 + 訂購成本
        fill_rate_results.append(fill_rate)
        total_costs.append(total_cost)

    return np.mean(fill_rate_results), np.mean(total_costs)

# 模擬兩種策略
fill_rate_check, cost_check = simulate_inventory(strategy="check")
print(f"檢查式策略 - Fill Rate: {fill_rate_check:.2f}, 總成本: {cost_check:.2f}")