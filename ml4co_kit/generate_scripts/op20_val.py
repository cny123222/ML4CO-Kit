from ..generator import OPDataGenerator
from ..utils.type_utils import SOLVER_TYPE

MAX_LENGTHS = {
    20: 2.,
    50: 3.,
    100: 4.
}

generator = OPDataGenerator(
    nodes_num=20, 
    max_length=MAX_LENGTHS[20], 
    data_type='dist',
    solver=SOLVER_TYPE.GUROBI,
    train_samples_num=0,
    val_samples_num=1280,
    test_samples_num=0,
    save_path="/mnt/nas-new/home/panwenzheng/chennuoyan/Dataset/Test/op",
    filename="op20_dist_1000"
)
generator.generate()