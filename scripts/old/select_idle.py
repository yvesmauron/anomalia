import pandas as pd
import os

idle_dict = {
    "data/resmed/staging/20200427/20200125_155702_0_HRD.edf.csv":{
        "type":"end",
        "start":"2020-01-25 18:05:00.0000",
        "end":None
    },
    "data/resmed/staging/20200427/20200124_120001_0_HRD.edf.csv":{
        "type":"end",
        "start":"2020-01-24 12:03:00.0000",
        "end":None
    }
}

for key, item in idle_dict.items():
    df = pd.read_csv(key)
    df.imeStamp = pd.to_datetime(df.TimeStamp, format="%Y-%m-%d %H:%M:%S.%f")
    
    if item['type'] == "end":
        train = df[df.TimeStamp > item["start"]]
        test = df[df.TimeStamp < item["start"]]
    elif item['type'] == "start":
        train = df[df.TimeStamp < item["end"]]
        test = df[df.TimeStamp > item["end"]]
    

    train_file = f"train_{os.path.basename(key)}"
    test_file = f"test_{os.path.basename(key)}"

    train.to_csv(os.path.join("data/resmed/staging", "BBett_idle", train_file))




