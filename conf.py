from pathlib import Path
import json
import os


template = {
    "station": "bia",
    "data_file": "exploration/video-analysis/data",
    "video": {
        "container": "videos",
        "file": "2020-07-27_Bianca_sd.mp4",
        "start_time_stamp": None,
        "end_time_stamp": None
    }
}

with open("./test.json", "r") as f:
    temp_1 = json.load(f)



path = Path("data/resmed/staging/20200907")

for p in path.iterdir():

    temp = template.copy()

    temp["data_file"] = os.path.join(temp["data_file"], os.path.basename(p))
    
    with open(f"./config/adls/{os.path.basename(p)[:15]}.json", "w") as f:
        json.dump(temp, f, indent=4)




