"""Operational Research App model

"""
from pathlib import Path
from enum import Enum, unique
from typing import NamedTuple, Tuple
import random

import numpy as np
from numpy.core.defchararray import array


@unique
class NeighborhoodType(Enum):
	
	RAND = 1
	HAM2 = 2,
	HAM3 = 3,
	HAM4 = 4

	def __str__(self) -> str:
		
		s = 'none'

		if self.value == self.RAND:
			s = 'Random'
		elif self.value == self.HAM2:
			s = 'Hamming 2'
		elif self.value == self.HAM3:
			s = 'Hamming 3'
		elif self.value == self.HAM4:
			s = 'Hamming 4'
		else:
			s = 'None'
		
		return s


@unique
class SolutionSelectrionMethod(Enum):

	BEST = 1,
	RANDOM = 2

	def __str__(self) -> str:
		if self.value == self.BEST:
			return 'Best'
		elif self.value == self.RANDOM:
			return 'Random'
		else:
			return 'None'


@unique
class Move(Enum):
	ONE_ZERO = 0,
	ZERO_ONE = 1
	

class CostFunctionParams(NamedTuple):
	alpha: float
	beta: float
	gamma: float

"""
class EvaluatedSolution(NamedTuple):
	solution: np.ndarray
	value: float
"""

class TabooList:

	def __init__(self):
		self._list: np.ndarray = np.array([])

	def contains(self, idx: int) -> bool:
		raise NotImplementedError

	def add(self, idx: int, mv: Move) -> None:
		raise NotImplementedError

	def remove(self, idx: int) -> None:
		raise NotImplementedError

	def at(self, idx: int) -> bool:
		raise NotImplementedError


class Model:

	def __init__(self):

		self.today: int = 0
		self.money: int = 0

		# matrices
		self._R: np.ndarray = np.array([])
		self._X: np.ndarray = np.array([])

		# vectors
		self._T: np.ndarray = np.array([])
		self._Q: np.ndarray = np.array([])
		self._E: np.ndarray = np.array([])
		self._P: np.ndarray = np.array([])

		# algorythm stuff
		#self.global_best: EvaluatedSolution
		self.global_best: Tuple[np.ndarray, float]
		self.iteration_limit: int
		self.aspiration_coefficient: float

		self.best_X_sequence: np.ndarray = np.array([])

		
	def taboo_search(self, max_iterations: int, nbrhd_type: NeighborhoodType, \
				ssm_type: SolutionSelectrionMethod, cutoff: float, nbrhd_size: int) -> Tuple[float, int]:
		raise NotImplementedError

	def load_data(self, filepath: Path) -> None:
		raise NotImplementedError
		
	def random_int(self) -> int:
		raise NotImplementedError

	def random_bool(self) -> bool:
		raise NotImplementedError

	def generate_initial(self) -> np.ndarray:
		raise NotImplementedError
		
	def _generate_new(self, nbrhd_type: NeighborhoodType, vec: np.ndarray) -> np.ndarray:
		raise NotImplementedError
		
	def generate_new_neighborhood(self, nbrhd_type: NeighborhoodType, vec: np.ndarray, size: int) -> np.ndarray:
		raise NotImplementedError
	
	def calculate_cost_function(self, vec: np.ndarray) -> float:
		raise NotImplementedError

	def determine_is_product_expired(self, size: int) -> bool:
		raise NotImplementedError
	
