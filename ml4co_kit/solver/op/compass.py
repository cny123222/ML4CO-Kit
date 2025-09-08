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

import itertools
import numpy as np
from multiprocessing import Pool
from typing import Union, Tuple, List
from ml4co_kit.solver.op.base import OPSolver
from ml4co_kit.utils.type_utils import SOLVER_TYPE
from ml4co_kit.utils.time_utils import iterative_execution, Timer


class OPCompassSolver(OPSolver):
    def __init__(
        self, 
        scale: int = 1e6, 
        time_limit: float = 120.0, 
        gurobi_gap: float = 0.0,
        precision: Union[np.float32, np.float64] = np.float32
    ):
        super(OPCompassSolver, self).__init__(
            solver_type=SOLVER_TYPE.GUROBI, scale=scale, precision=precision
        )
        self.time_limit = time_limit
        self.gurobi_gap = gurobi_gap
        
    def _solve(
        self, 
        depots: np.ndarray, 
        points: np.ndarray, 
        prizes: np.ndarray, 
        max_length: float
    ) -> Tuple[float, List[int]]:
        """
        Solve a single OP instance using Compass
        """

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

    def __str__(self) -> str:
        return "OPGurobiSolver"