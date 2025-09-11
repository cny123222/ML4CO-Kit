import os
from tqdm import tqdm
from .. import PCTSPILSSolver

output_dir = "/mnt/nas-new/home/panwenzheng/chennuoyan/Dataset/Train/pctsp/"
os.makedirs(output_dir, exist_ok=True)

for seed in tqdm(range(1, 16)):
    print(f"[DEBUG] Processing seed {seed}")
    solver = PCTSPILSSolver()
    
    input_file = f"/mnt/nas-new/home/panwenzheng/chennuoyan/CORectifier/attention-learn-to-route/data/pctsp/pctsp100_train_seed{seed}.pkl"
    output_file = os.path.join(output_dir, f"pctsp100_train_seed{seed}.txt")

    solver.from_pkl(input_file)
    print("[DEBUG] Loaded data from pickle")
    solver.solve(num_threads=64, show_time=True)
    print("[DEBUG] Solved the problem")
    print(solver.evaluate(calculate_gap=False))
    solver.to_txt(output_file)
    
print("[INFO] All individual seed files have been generated.")
print("[INFO] Now merging them into a single file...")

final_output_file = os.path.join(output_dir, "pctsp100_train.txt")

with open(final_output_file, 'w') as outfile:
    for seed in range(16):
        seed_file_path = os.path.join(output_dir, f"pctsp100_train_seed{seed}.txt")
        if os.path.exists(seed_file_path):
            with open(seed_file_path, 'r') as infile:
                content = infile.read()
                outfile.write(content)

print(f"[INFO] Successfully merged all files into: {final_output_file}")


print("[INFO] Cleaning up temporary seed files...")
cleanup_successful = True
for seed in range(16):
    seed_file_path = os.path.join(output_dir, f"pctsp100_train_seed{seed}.txt")
    try:
        if os.path.exists(seed_file_path):
            os.remove(seed_file_path)
    except OSError as e:
        print(f"Error removing file {seed_file_path}: {e}")
        cleanup_successful = False

if cleanup_successful:
    print("[INFO] Cleanup complete. All temporary files removed.")
else:
    print("[WARNING] Some temporary files could not be removed.")