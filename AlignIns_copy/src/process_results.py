import pandas as pd
import os
import re


def add_file(results_file, file_to_add):
    
    

    best_block = False
    clean_acc = asr = rob_acc = None
    with open(file_to_add, "r", encoding = "utf-8",errors = "ignore") as f:
        for line in f:
            if "Best results" in line:
                best_block = True
                continue
            if best_block:
                if "Clean ACC" in line:
                    clean_acc = float(re.search(r"([0-9.]+)\s*$", line).group(1))
                elif "Attack Success Ratio" in line:
                    asr = float(re.search(r"([0-9.]+)\s*$", line).group(1))
                elif "Robustness ACC" in line:
                    rob_acc = float(re.search(r"([0-9.]+)\s*$", line).group(1))
    
    tmp_df = pd.DataFrame([{
        "file_name": file_to_add,
        "clean_acc": clean_acc,
        "asr": asr,
        "rob_acc": rob_acc
    }])
    print(tmp_df.head())

    if os.path.exists(results_file):
        df = pd.read_excel(results_file)
        df = pd.concat([df,tmp_df], ignore_index = True)
        
    else:
        df = tmp_df.copy()
    df.to_excel(results_file,index = False)


def add_file_2(results_file, file_to_add):
    
    

    best_block = False
    clean_acc = asr = rob_acc = None
    with open(file_to_add, "r", encoding = "utf-8",errors = "ignore") as f:
        for line in f:
            if "Best results" in line:
                best_block = True
                continue
            if best_block:
                if "Clean ACC" in line:
                    clean_acc = float(re.search(r"([0-9.]+)\s*$", line).group(1))
                elif "Attack Success Ratio" in line:
                    asr = float(re.search(r"([0-9.]+)\s*$", line).group(1))
                elif "Robustness ACC" in line:
                    rob_acc = float(re.search(r"([0-9.]+)\s*$", line).group(1))
    
    _,dataset,_,server_defense, fedserver_defenes, attack,_,num_attackers = file_to_add.split(".log")[0].split("_")
    tmp_df = pd.DataFrame([{
        "dataset": dataset,
        "server_defense":server_defense,
        "feserver_defense":fedserver_defenes,
        "attack":attack,
        "num_attackers":num_attackers,
        "clean_acc": clean_acc,
        "asr": asr,
        "rob_acc": rob_acc
    }])
    print(tmp_df.head())

    if os.path.exists(results_file):
        df = pd.read_excel(results_file)
        df = pd.concat([df,tmp_df], ignore_index = True)
        
    else:
        df = tmp_df.copy()
    df.to_excel(results_file,index = False)


def proces_xlsx(file):
    
    df = pd.read_excel(file)
    # columns = df.loc[0]
    # df.columns = columns
    # df.drop(["Unnamed: 3", "file_name"], axis= 1, inplace = True)
    df.sort_values(by = ["dataset", "attack"], inplace = True)
    for col in ["clean_acc", "asr", "rob_acc"]:  # put your float columns here
        df[col] = df[col].astype(float)*100
    print(df.head())
    print(df.columns)
    df.to_excel(file,index=False, float_format="%.4f")

import os
if __name__=="__main__":
    

    
    # add_file("results.xlsx","sfl_cifar10_avg_no_attack.log")
    # add_file("results.xlsx","sfl_fmnist_avg_badnet_attack.log")
    # add_file("results.xlsx","sfl_fmnist_alignins_badnet_attack.log")

    # add_file("results.xlsx","sfl_cifar10_avg_badnet_attack.log")
    # add_file("results.xlsx","sfl_cifar10_alignins_badnet_attack.log")
    # add_file("results.xlsx","sfl_fmnist_avg_dba_attack.log")

    # add_file("results.xlsx","sfl_fmnist_alignins_dba_attack.log")
    # add_file("results.xlsx","sfl_cifar10_avg_dba_attack.log")
    # add_file("results.xlsx","sfl_cifar10_alignins_dba_attack.log")

    # add_file("results.xlsx","sfl_fmnist_avg_neurotoxin_attack.log")
    # add_file("results.xlsx","sfl_fmnist_alignins_neurotoxin_attack.log")
    # add_file("results.xlsx","sfl_cifar10_avg_neurotoxin_attack.log")

    # add_file("results.xlsx","sfl_cifar10_alignins_neurotoxin_attack.log")
    # add_file("results.xlsx","sfl_mnist_alignins_dba_attack.log")
    # add_file("results.xlsx","sfl_mnist_alignins_neurotoxin_attack.log")
    # add_file("results.xlsx","sfl_mnist_avg_no_attack_2.log")
    # files = [f for f in os.listdir(".") if f.endswith("_1.log") or f.endswith("_4.log")]
    # print(files)
    # for file in files:
    #     add_file_2("results_5.xlsx",file)
    proces_xlsx("../logs/results_2.xlsx")