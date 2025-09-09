r"""
Compass Solver for solving OP.
"""

# Copyright (c) 2024 Thinklab@SJTU
# ML4CO-Kit is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
# http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import os
import tempfile
import shutil
import numpy as np
from multiprocessing import Pool
from typing import Union, Tuple, List
from subprocess import check_call, check_output
from ml4co_kit.solver.op.base import OPSolver
from ml4co_kit.utils.type_utils import SOLVER_TYPE
from ml4co_kit.utils.time_utils import iterative_execution, Timer

call_cnt = 0

class OPCompassSolver(OPSolver):
    def __init__(
        self, 
        scale: int = 1e7, 
        executable: str = None,
        precision: Union[np.float32, np.float64] = np.float64
    ):
        super(OPCompassSolver, self).__init__(
            solver_type=SOLVER_TYPE.COMPASS, scale=scale, precision=precision
        )
        self.executable = executable if executable is not None else shutil.which("compass")
        assert self.executable is not None, (
            "Could not find the 'compass' executable in the system's PATH.\n"
            "Please ensure that Compass is installed correctly and that the directory "
            "containing the 'compass' executable (e.g., /usr/local/bin) is in the PATH environment variable."
        )
        
    def _solve(
        self, 
        depot: np.ndarray, 
        points: np.ndarray, 
        prizes: np.ndarray, 
        max_length: float,
        name: str = "problem"
    ) -> Tuple[float, List[int]]:
        """
        Solve a single OP instance using Compass
        """   
        
        # with tempfile.TemporaryDirectory() as tempdir:
        #     problem_filename = os.path.join(tempdir, f"{name}.oplib")
        #     tour_filename = os.path.join(tempdir, f"{name}.tour")
        #     log_filename = os.path.join(tempdir, f"{name}.log")

        #     try:
        #         self._write_oplib(problem_filename, depot, points, prizes, max_length)

        #         with open(log_filename, 'w') as f:
        #             check_call([self.executable, '--op', '--op-ea4op', problem_filename, '-o', tour_filename],
        #                     stdout=f, stderr=f)

        #         tour = self._read_oplib(tour_filename, n=len(prizes))
        #         tour_length = self._calc_op_length(depot, points, tour)
        #         if not tour_length <= max_length:
        #             print("Warning: length exceeds max length:", tour_length, max_length)
        #         assert tour_length <= max_length + 1e-5, "Tour exceeds max_length!"
        #         return tour

        #     except Exception as e:
        #         print("Exception occured")
        #         print(e)
        #         return None
        
        global call_cnt
        
        debug_dir = os.path.join(os.getcwd(), "debug")
        os.makedirs(debug_dir, exist_ok=True)
    
        problem_filename = os.path.join(debug_dir, f"{call_cnt}.oplib")
        tour_filename = os.path.join(debug_dir, f"{call_cnt}.tour")
        log_filename = os.path.join(debug_dir, f"{call_cnt}.log")
        
        call_cnt += 1
        
        depot = depot.tolist()
        points = points.tolist()
        prizes = prizes.tolist()

        try:
            self._write_oplib(problem_filename, depot, points, prizes, max_length)

            with open(log_filename, 'w') as f:
                check_call([self.executable, '--op', '--op-ea4op', problem_filename, '-o', tour_filename],
                        stdout=f, stderr=f)

            tour = self._read_oplib(tour_filename, n=len(prizes))
            tour_length = self._calc_op_length(depot, points, tour)
            if not tour_length <= max_length:
                print("Warning: length exceeds max length:", tour_length, max_length)
            assert tour_length <= max_length + 1e-5, "Tour exceeds max_length!"
            return tour

        except Exception as e:
            print("Exception occured")
            print(e)
            return None

    def solve(
        self, 
        depots: Union[list, np.ndarray] = None,
        points: Union[list, np.ndarray] = None,
        prizes: Union[list, np.ndarray] = None,
        max_lengths: Union[list, np.ndarray] = None,
        num_threads: int = 1, 
        show_time: bool = False, 
    ) -> np.ndarray:
        """
        Solve OP instances using Compass
        """
        # preparation
        self.from_data(
            depots=depots, points=points, prizes=prizes, max_lengths=max_lengths
        )
        timer = Timer(apply=show_time)
        timer.start()

        # solve
        tours = list()
        num_points = self.points.shape[0]
        if num_threads == 1:
            for idx in iterative_execution(range, num_points, self.solve_msg, show_time):
                tour = self._solve(
                    self.depots[idx],
                    self.points[idx],
                    self.prizes[idx],
                    self.max_lengths[idx]
                )
                tours.append(tour)
        else:
            for idx in iterative_execution(
                range, num_points // num_threads, self.solve_msg, show_time
            ):
                with Pool(num_threads) as p1:
                    cur_tours = p1.starmap(
                        self._solve,
                        [
                            (self.depots[idx*num_threads+inner_idx],
                             self.points[idx*num_threads+inner_idx],
                             self.prizes[idx*num_threads+inner_idx],
                             self.max_lengths[idx*num_threads+inner_idx])
                            for inner_idx in range(num_threads)
                        ],
                    )
                for tour in cur_tours:
                    tours.append(tour)

        # format
        self.from_data(tours=tours, ref=False)
        
        # show time
        timer.end()
        timer.show_time()
        
        # return
        return self.tours
    
    def _write_oplib(self, filename, depot, loc, prize, max_length, name="problem"):
        
        # print("[DEBUG] In _write_oplib")
        # print(filename, depot, loc, prize, max_length)
        
        with open(filename, 'w') as f:
            f.write("\n".join([
                "{} : {}".format(k, v)
                for k, v in (
                    ("NAME", name),
                    ("TYPE", "OP"),
                    ("DIMENSION", len(loc) + 1),
                    ("COST_LIMIT", int(max_length * 10000000 + 0.5)),
                    ("EDGE_WEIGHT_TYPE", "EUC_2D"),
                )
            ]))
            f.write("\n")
            f.write("NODE_COORD_SECTION\n")
            f.write("\n".join([
                "{}\t{}\t{}".format(i + 1, int(x * 10000000 + 0.5), int(y * 10000000 + 0.5))  # oplib does not take floats
                #"{}\t{}\t{}".format(i + 1, x, y)
                for i, (x, y) in enumerate([depot] + loc)
            ]))
            f.write("\n")
            f.write("NODE_SCORE_SECTION\n")
            f.write("\n".join([
                "{}\t{}".format(i + 1, d)
                for i, d in enumerate([0] + prize)
            ]))
            f.write("\n")
            f.write("DEPOT_SECTION\n")
            f.write("1\n")
            f.write("-1\n")
            f.write("EOF\n")
            
    def _read_oplib(self, filename, n):
        with open(filename, 'r') as f:
            tour = []
            dimension = 0
            started = False
            for line in f:
                if started:
                    loc = int(line)
                    if loc == -1:
                        break
                    tour.append(loc)
                if line.startswith("DIMENSION"):
                    dimension = int(line.split(" ")[-1])

                if line.startswith("NODE_SEQUENCE_SECTION"):
                    started = True
        
        assert len(tour) > 0, "Unexpected length"
        tour = np.array(tour).astype(int) - 1  # Subtract 1 as depot is 1 and should be 0
        assert tour[0] == 0  # Tour should start with depot
        assert tour[-1] != 0  # Tour should not end with depot
        return tour[1:].tolist()
    
    def _write_compass_par(self, filename, parameters):
        default_parameters = {  # Use none to include as flag instead of kv
            "SPECIAL": None,
            "MAX_TRIALS": 10000,
            "RUNS": 10,
            "TRACE_LEVEL": 1,
            "SEED": 0
        }
        with open(filename, 'w') as f:
            for k, v in {**default_parameters, **parameters}.items():
                if v is None:
                    f.write("{}\n".format(k))
                else:
                    f.write("{} = {}\n".format(k, v))
    
    def _calc_op_length(self, depot, loc, tour):
        assert len(np.unique(tour)) == len(tour), "Tour cannot contain duplicates"
        loc_with_depot = np.vstack((np.array(depot)[None, :], np.array(loc)))
        sorted_locs = loc_with_depot[np.concatenate(([0], tour, [0]))]
        return np.linalg.norm(sorted_locs[1:] - sorted_locs[:-1], axis=-1).sum()
            
    def __str__(self) -> str:
        return "OPCompassSolver"