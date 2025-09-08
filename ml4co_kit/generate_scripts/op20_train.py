from ..generator import OPDataGenerator
from ml4co_kit.utils.type_utils import SOLVER_TYPE

MAX_LENGTHS = {
    20: 2.,
    50: 3.,
    100: 4.
}

for i in range(1, 11):
    generator = OPDataGenerator(
        nodes_num=20, 
        max_length=MAX_LENGTHS[20], 
        data_type='dist',
        solver=SOLVER_TYPE.GUROBI,
        train_samples_num=128000,
        val_samples_num=0,
        test_samples_num=0,
        save_path="/mnt/nas-new/home/panwenzheng/chennuoyan/Dataset/Train/op/op20_dist_gurobi",
        filename=f"op20_dist_128k_{i}"
    )
    generator.generate()