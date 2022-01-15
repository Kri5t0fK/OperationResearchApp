"""Operational Research App model

"""
from pathlib import Path # universal Path
from enum import Enum, unique # helpful
from typing import NamedTuple, Tuple # code visibility
import random # generating initial solutions
import json # data parsing

import numpy as np

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
	

class CostFunctionParams(NamedTuple):
	alpha: float
	beta: float
	gamma: float

"""
class EvaluatedSolution(NamedTuple):
	solution: np.ndarray
	value: float
"""

class TabuList:

	def __init__(self):
		self._list: np.ndarray = np.array([])

	def contains(self, idx: int) -> bool:
		raise NotImplementedError

	def add(self, idx: int) -> None:
		# add just recipe-idx, that shouldn't be chosen
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

		# vectors
		self._T: np.ndarray = np.array([])
		self._Q: np.ndarray = np.array([])
		self._E: np.ndarray = np.array([])
		self._P: np.ndarray = np.array([])
		self._X: np.ndarray = np.array([])	# current X (solution), stores selected R_IDXs (vec of ints, not bools)

		# problem dimensions R: N_Rec x N_Ing
		self._N_Rec: int = 0	# N_Rec -> number of recipies
		self._N_Ing: int = 0	# N_Ing -> number of ingredients

		# algorythm stuff
		self.recipe_count: int						# target recipe count (length of _X)
		self.initial_X: Tuple[float, np.ndarray]	# [cost_function_val, _X]
		#self.global_best: EvaluatedSolution
		self.global_best: Tuple[int, float]
		self.iteration_limit: int
		self.aspiration_coefficient: float
		self.tabu_age: np.ndarray = np.array([0, 0, 0])	# [short, medium, long]

		# cost function stuff
		self.params: np.ndarray = np.array([0.0, 0.0, 0.0])

		random.seed() # seed the random number generator with system time (pretty random)

		
	def tabu_search(self, max_iterations: int, nbrhd_type: NeighborhoodType, \
				ssm_type: SolutionSelectrionMethod, cutoff: float, nbrhd_size: int) -> Tuple[float, int]:
		raise NotImplementedError


	def load_data(self, filepath: Path) -> None:
		if not filepath.exists():
			raise ValueError('File doesn\'t exsist')
		
		with filepath.open() as json_f:
			model_data = json.load(json_f)

			self._N_Rec = model_data['n']
			self._N_Ing = model_data['m']

			self.today = 1
			self.money = model_data['money']

			self._R = model_data['recipies']
			self._T = model_data['times']
			self._Q = model_data['quantities']
			self._E = model_data['dates']
			self._P = model_data['prices']


	def set_params(self, a: float, b: float, g:float) -> None:
		self.params = np.array([a, b, g])


	def set_tabu_age(self, short: int, medium: int, long: int) -> None:
		self.tabu_age = np.array([short, medium, long])


	def generate_initial(self) -> None:
		new_X: np.ndarray = np.array([], dtype=int)
		for _ in range(self.recipe_count):
			curr_recipe = self.random_recipe_idx()
			while curr_recipe in new_X:
				curr_recipe = self.random_recipe_idx()
			new_X = np.append(new_X, [curr_recipe])
		self.initial_X = np.array([self.calculate_cost_function(new_X), new_X])


	def generate_new(self, nbrhd_type: NeighborhoodType, vec: np.ndarray) -> np.ndarray:
		raise NotImplementedError


	def generate_new_neighborhood(self, nbrhd_type: NeighborhoodType, vec: np.ndarray) -> np.ndarray:
		raise NotImplementedError


	def calculate_cost_function(self, vec_X: np.ndarray) -> float:
		cost: np.ndarray = np.array([0.0, 0.0, 0.0]) # Shop, Loss, Time
		q_need: np.ndarray = np.zeros(self._N_Ing) # needed quantity of each Ingredient

		# j: recipe_idx
		for j in vec_X:
			cost[2] += self._T[j]	# add _R[j]'s time to Time-cost
			for i in range(self._N_Ing):
				q_need += self._R[j, i]	# calc need for each Ingredient
		
		# calc Shop & Loss costs
		for i in range(self._N_Ing):
			#TODO: maybe write the [q_need[i]-self._Q[i]] to aux var and then determine to which cost to add
			cost[0] += self._P[i] * max(0, q_need[i] - self._Q[i])
			if self.determine_is_product_expired(i):
				cost[1] += self._P[i] * max(0, self._Q[i] - q_need[i])

		return self.params @ cost


	def determine_is_product_expired(self, idx: int) -> bool:
		return self._E[idx] < self.today + 1


	def random_recipe_idx(self) -> int:
		# TODO: check out random.shuffle(x) -> shuffles list `x` randomly
		return random.randrange(start=0, stop=self._N_Rec, step=1)

	def random_bool(self) -> bool:
		rand_int = random.randrange(start=0, stop=100, step=1)
		return rand_int < 50 # true if less than 50, false if greater


	@staticmethod
	def max(a, b):
		if a > b:
			return a
		return b
