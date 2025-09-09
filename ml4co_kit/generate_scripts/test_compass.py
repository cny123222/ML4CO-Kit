from .. import OPDataGenerator, OPCompassSolver
from ..utils.type_utils import SOLVER_TYPE

# generator = OPDataGenerator(
#     nodes_num=20, 
#     max_length=2., 
#     data_type='dist',
#     solver=SOLVER_TYPE.GUROBI,
#     train_samples_num=0,
#     val_samples_num=5,
#     test_samples_num=0,
#     save_path="/mnt/nas-new/home/panwenzheng/chennuoyan/Dataset/Test/op",
#     filename="op20_dist_test"
# )
# generator.generate()

solver = OPCompassSolver(executable="/mnt/nas-new/home/panwenzheng/chennuoyan/compass/compass")
# solver.from_txt("/mnt/nas-new/home/panwenzheng/chennuoyan/Dataset/Test/op/op20_dist_test.txt", ref=True)
solver.from_pickle("/mnt/nas-new/home/panwenzheng/chennuoyan/CORectifier/attention-learn-to-route/data/op/op_dist20_test_run_seed1234.pkl")
solver.solve()
# print(solver.evaluate(calculate_gap=False))