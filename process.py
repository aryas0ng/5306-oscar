import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import os

us_temp_dataset = [[]] * 10
world_temp_dataset = [[]] * 10
data_path = "./data/"
for i in os.listdir(data_path):
    if i.endswith(".csv"):
        strings = i.split("_")
        year = int(strings[0])
        # print(year)
        loc = strings[1][:-5]
        # print(loc)
        num = strings[1][-5:-4]
        # print(num)
        data = pd.read_csv("./data/"+i,skiprows=2).iloc[:,1:]
        colnames = data.columns.values
        for i in range(len(colnames)):
            colnames[i] = colnames[i].split(" oscar")[0]
        if loc=="us":
            if len(us_temp_dataset[year-2015]) == 0:
                us_temp_dataset[year-2015]=us_temp_dataset[year-2015] + [data]
            else:
                us_temp_dataset[year-2015]=us_temp_dataset[year-2015] + [data.iloc[:,1:]]
        else:
            if len(world_temp_dataset[year-2015]) == 0:
                world_temp_dataset[year-2015]=world_temp_dataset[year-2015] + [data]
            else:
                world_temp_dataset[year-2015]=world_temp_dataset[year-2015] + [data.iloc[:,1:]]

us_dataset = []
world_dataset = []
for year in range(10):
    us_year_temp_data = us_temp_dataset[year]
    world_year_temp_data = world_temp_dataset[year]
    us_dataset.append(pd.concat(us_year_temp_data,axis=1))
    world_dataset.append(pd.concat(world_year_temp_data,axis=1))

winners = ["BIRDMAN","SPOTLIGHT","MOONLIGHT","THE SHAPE OF WATER","GREEN BOOK","PARASITE","NOMADLAND","CODA",
           "EVERYTHING EVERYWHERE ALL AT ONCE","OPPENHEIMER"]

winner_idx = [0, 4, 1, 0, 4, 3, 1, 1, 1, 8]

# 参考框架（复制用不要直接用）：
# for i in range(len(us_dataset)):
#     print(str(2015+i)+"---US------------------------------------------")
#     print(us_dataset[i])
#     print(str(2015+i)+"---WORLD------------------------------------------")
    # print(world_dataset[i])
#     print(str(2015+i)+"---WINNER------------------------------------------")
#     print(winners[i])

# Return the ranking of winner based on the maximum value, 0-based ranking
def maximum(df, idx):
    max_value = df.max()
    sorted = max_value.sort_values().rank(ascending=False)
    return sorted[winners[idx]]-1

def mean(df, idx):
    mean = df.mean()
    print(mean)
    sorted = mean.sort_values().rank(ascending=False)
    print(sorted)
    return sorted[winners[idx]]-1

# Return the ranking of the winner based on exponential smoothing and prediction, 0-based ranking
def exponential_smoothing(df, alpha, idx):
    smoothed = df.ewm(alpha=alpha).mean()
    pred = smoothed.iloc[-1] + alpha * (df.iloc[-1] - smoothed.iloc[-1])
    sorted = pred.sort_values().rank(ascending=False)
    return sorted[winners[idx]]-1

def sarima_helper(df, order, seasonal_order, steps):
    model = SARIMAX(df, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    model_fit = model.fit(disp=False)
    forecast = model_fit.forecast(steps=steps)
    return forecast

def sarima(df, order = (1,0,0), seasonal_order = (0,0,0,1), steps = 1):
    forecasts = {}
    for col in df.columns:
        series = df[col]
        forecast = sarima_helper(series, order, seasonal_order, steps)
        forecasts[col] = forecast
    return forecasts

for i in range(len(us_dataset)):
    print(str(2015+i)+"---Maximum------------------------------------------")
    print(str(2015+i)+"---US------------------------------------------")
    max_us = maximum(us_dataset[i], i)
    print(max_us)
    print(str(2015+i)+"---WORLD------------------------------------------")
    max_world = maximum(world_dataset[i], i)
    print(max_world)

    print(str(2015+i)+"---Mean------------------------------------------")
    print(str(2015+i)+"---US------------------------------------------")
    mean_us = mean(us_dataset[i], i)
    print(mean_us)
    print(str(2015+i)+"---WORLD------------------------------------------")
    mean_world = mean(world_dataset[i], i)
    print(mean_world)

    print(str(2015+i)+"---Exp Smoothing------------------------------------------")
    alpha = 0.5
    print(str(2015+i)+"---US------------------------------------------")
    exp_us = exponential_smoothing(us_dataset[i], alpha, i)
    print(exp_us)
    print(str(2015+i)+"---WORLD------------------------------------------")
    exp_wolrd = exponential_smoothing(world_dataset[i], alpha, i)
    print(exp_wolrd)

    print(str(2015+i)+"---SARIMA------------------------------------------")
    order = (1,1,1)
    seasonal_order = (1,1,1,5)
    steps = 1
    print(str(2015+i)+"---US------------------------------------------")
    sarima_us = sarima(us_dataset[i], order, seasonal_order, steps)
    for col, forecast in sarima_us.items():
        print(f"Movie {col}: {forecast}")
    print(str(2015+i)+"---WORLD------------------------------------------")
    sarima_world = sarima(world_dataset[i], order, seasonal_order, steps)
    for col, forecast in sarima_world.items():
        print(f"Movie {col}: {forecast}")
