import os

scripts = [
    "process_wandb.py",
    "process_final_results.py",
    "split_results_by_dataset.py",
    "process_combine_results.py"
]

for script in scripts:
    print(f"Running {script}...")
    ret = os.system(f"python {script}")
    
    if ret != 0:
        print(f"❌ {script} failed. Stopping.")
        break