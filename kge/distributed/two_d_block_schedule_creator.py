import random
import numpy as np
from typing import List, Tuple


class TwoDBlockScheduleCreator:
    """
    Creates a non blocking schedule for TwoDBLock partitioning
    can only handle num partitions of base 2
    num partitions needs to be 2*num_workers
    """
    def __init__(self, num_partitions, num_workers, randomize_iterations=False):
        self.num_partitions = num_partitions
        self.num_workers = num_workers
        self.randomize_iterations = randomize_iterations

    def create_schedule(self) -> List[List[Tuple[int, int]]]:
        """
        creates non blocking schedule
        Returns:
            list of iterations
            each iteration is a list of blocks (i,j) of size num_workers
        """
        schedule = []
        schedule.extend(self._create_schedule(self.num_partitions, self.num_workers))
        if self.randomize_iterations:
            random.shuffle(schedule)
        return schedule

    def _create_schedule(self, n_p, n_w, offset=0) -> List[List[Tuple[int, int]]]:
        schedule = []
        if n_p == 2 and n_w == 1:
            schedule.extend(self._handle_2x2_diagonal(offset=offset))
            return schedule
        else:
            # anti diagonal upper right quadrant
            schedule.extend(
                self._handle_anti_diagonal(
                    int(n_p / 2), x_offset=offset + int(n_p / 2), y_offset=offset
                )
            )
            # anti diagonal lower left quadrant
            schedule.extend(
                self._handle_anti_diagonal(
                    int(n_p / 2), x_offset=offset, y_offset=offset + int(n_p / 2)
                )
            )
            # both diagonal blocks
            schedule.extend(
                self._concat_schedules(
                    self._create_schedule(int(n_p / 2), int(n_w / 2), offset=offset),
                    self._create_schedule(
                        int(n_p / 2), int(n_w / 2), offset=offset + int(n_p / 2)
                    ),
                )
            )
            return schedule

    @staticmethod
    def _handle_2x2_diagonal(offset=0) -> List[List[Tuple[int, int]]]:
        """
        handle smallest diagonal block
        2x2 with one worker
        Args:
            offset: position in the complete diagonal

        Returns:
            List of iterations for one worker
        """
        schedule = list()
        for i in range(2):
            for j in range(2):
                schedule.append([(i + offset, j + offset)])
        random.shuffle(schedule)
        return schedule

    @staticmethod
    def _handle_anti_diagonal(n_p, x_offset=0, y_offset=0):
        permutation_matrix = np.random.permutation(np.diag(np.ones(n_p, dtype=np.int)))
        schedule = []
        for i in range(int(n_p)):
            iteration = list(zip(*permutation_matrix.nonzero()))
            for j, block in enumerate(iteration):
                block = ((block[0] + i) % (int(n_p)) + x_offset, block[1] + y_offset)
                iteration[j] = block
            schedule.append(iteration)
        return schedule

    @staticmethod
    def _concat_schedules(schedule_1, schedule_2):
        schedule = list()
        for iteration_1, iteration_2 in zip(schedule_1, schedule_2):
            iteration_1.extend(iteration_2)
            schedule.append(iteration_1)
        return schedule