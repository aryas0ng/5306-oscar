import pandas as pd
import numpy as np
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
# 参考框架（复制用不要直接用）：
for i in range(len(us_dataset)):
    print(str(2015+i)+"---US------------------------------------------")
    print(us_dataset[i])
    print(str(2015+i)+"---WORLD------------------------------------------")
    print(world_dataset[i])
    print(str(2015+i)+"---WINNER------------------------------------------")
    print(winners[i])