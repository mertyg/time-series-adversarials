import os
import pandas as pd
import json

base_dir = "/om2/user/merty/dummyresult/t/"

results = []
for ds in os.listdir(base_dir):
    if ds == "dataset":
        continue
    ds_folder = os.path.join(base_dir, ds)
    for model in os.listdir(ds_folder):
        model_folder = os.path.join(ds_folder, model)
        for attack_dir in os.listdir(model_folder):
            attack_folder = os.path.join(model_folder, attack_dir)
            if not os.path.isdir(attack_folder):
                continue
            attack_eps = attack_dir.split("_")[-1]
            train_file = os.path.join(attack_folder, "train_result.json")
            val_file = os.path.join(attack_folder, "val_result.json")
            res = {}
            res["Attack"] = attack_dir
            res["Model"] = model
            res["Dataset"] = ds
            with open(train_file, "r") as fp:
                train_dict = json.load(fp)
                res["TrainRegularAccuracy"] = train_dict["0"]
                res["TrainAdversarialAccuracy"] = train_dict[attack_eps]
            with open(val_file, "r") as fp:
                val_dict = json.load(fp)
                res["ValRegularAccuracy"] = val_dict["0"]
                res["ValAdversarialAccuracy"] = val_dict[attack_eps]
            results.append(res)

pd.DataFrame.from_records(results).to_csv("./timeseries.csv")
