from .. import PCTSPORSolver

solver = PCTSPORSolver(time_limit=60)
solver.from_pickle("/mnt/nas-new/home/panwenzheng/chennuoyan/CORectifier/attention-learn-to-route/data/pctsp/pctsp20_train_seed42.pkl")
print("[DEBUG] Loaded data from pickle")
solver.solve(num_threads=64, show_time=True)
print("[DEBUG] Solved the problem")
print(solver.evaluate(calculate_gap=False))
solver.to_txt("/mnt/nas-new/home/panwenzheng/chennuoyan/Dataset/Train/pctsp/pctsp20_train.txt")