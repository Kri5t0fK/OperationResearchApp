"""Operational Research App model

"""
from pathlib import Path # universal Path
from enum import Enum, unique # helpful
from typing import NamedTuple, Tuple, List # code visibility
import random # generating initial solutions
import json # data parsing

import numpy as np

@unique
class NeighborhoodType(Enum):
	
	RAND = 0
	HAM1 = 1
	HAM2 = 2
	HAM3 = 3
	HAM4 = 4
	HAM5 = 5
	HAM6 = 6

	def __str__(self) -> str:
		if self.value == self.RAND.value:
			return 'Random'
		elif self.value > self.RAND.value:
			return f'Hamming {self.value}'
		else:
			return 'None'


@unique
class SolutionSelectionMethod(Enum):

	BEST = 1,
	RANDOM = 2

	def __str__(self) -> str:
		if self.value == self.BEST.value:
			return 'Best'
		elif self.value == self.RANDOM.value:
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

"""
class TabuList:

	def __init__(self):
		self._list: np.ndarray = np.array([])

	def contains(self, idx: int) -> bool:
		return idx in self._list

	def append(self, idx: int) -> None:
		# add just recipe-idx, that shouldn't be chosen
		self._list = np.append(self._list, [idx])

	def pop(self, idx: int) -> None:
		self._list = np.delete(self._list, 0)

	def at(self, idx: int) -> bool:
		raise NotImplementedError
"""

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
		# [cost_function, current X (solution)], stores selected R_IDXs (vec of ints, not bools)
		self._X: Tuple[float, np.ndarray]

		# problem dimensions R: N_Rec x N_Ing
		self._N_Rec: int = 0	# N_Rec -> number of recipies
		self._N_Ing: int = 0	# N_Ing -> number of ingredients

		# algorythm stuff
		self.recipe_count: int						# target recipe count (length of _X)
		self.initial_X: Tuple[float, np.ndarray]	# [cost_function_val, _X]
		#self.global_best: EvaluatedSolution
		self.global_best_X: Tuple[int, float, np.ndarray]	# [iteration, cost_function_val, _X]
		self.iteration_limit: int
		self.neighborhood_size: int
		self.aspiration_coeff: float
		self.tabu_age: np.ndarray = np.array([0, 0, 0])	# [short, medium, long]

		# cost function stuff
		self.params: np.ndarray = np.array([0.0, 0.0, 0.0])	# [alpha, beta, gamma]

		# data for gui graph of cost_function
		self.graph_data: np.ndarray[np.ndarray, np.ndarray]	# [current_cost, best_cost]

		random.seed() # seed the random number generator with system time (pretty random)

		
	def tabu_search(self, max_iterations: int, nbrhd_hamming: NeighborhoodType, \
				ssm_type: SolutionSelectionMethod, cutoff: float, nbrhd_size: int) -> Tuple[int, float, np.ndarray]:
		
		tabu_list_short: np.ndarray = np.array([], dtype=int)
		self.global_best_X = (0, self.initial_X[0], self.initial_X[1].copy())	# best solution
		self._X = (self.initial_X[0], self.initial_X[1].copy())					# current solution

		# initialize graph data
		self.graph_data = np.array([[self.initial_X[0]], [self.initial_X[0]]])

		iteration: int = 1
		while iteration < max_iterations:
			neighborhood: np.ndarray = self.generate_new_neighborhood(self._X[1], nbrhd_hamming)

			# pick only neighbors [not forbidden by Tabu] or [meeting aspiration]
			neighborhood_eligible: np.ndarray = np.array([[]])
			for candidate in neighborhood:
				# check Tabu eligibility
				eligible: bool = True
				for recipe in candidate:
					if recipe in tabu_list_short:
						eligible = False

				# aspiration check
				if not eligible:
					if self.calculate_cost_function(candidate) < self.aspiration_coeff * self.global_best_X[1]:
						eligible = True
				
				# add to eligible neighborhood
				if eligible:
					if neighborhood_eligible.size == 0:
						neighborhood_eligible = np.append(neighborhood_eligible, [candidate], 1)
					else:
						neighborhood_eligible = np.append(neighborhood_eligible, [candidate], 0)
			
			#TODO: what if neighborhood_eligible empty? pick from [tabu_medium]? - for now forces "bad/forbidden" solution
			if neighborhood_eligible.size == 0:
				neighborhood_eligible = np.append(neighborhood_eligible, [neighborhood[0]], 1)

			# select new neighbor
			new_neighbor: np.ndarray = neighborhood_eligible[0]
			if ssm_type == SolutionSelectionMethod.RANDOM:
				# randomly
				new_neighbor = random.choice(neighborhood_eligible)
			elif ssm_type == SolutionSelectionMethod.BEST:
				# best
				new_neighbor_cost: float = self.calculate_cost_function(new_neighbor)
				for candidate in neighborhood_eligible:
					candidate_cost: float = self.calculate_cost_function(candidate)
					if candidate_cost < new_neighbor_cost:
						new_neighbor = candidate
						new_neighbor_cost = candidate_cost
			
			# check if new best
			new_neighbor_cost: float = self.calculate_cost_function(new_neighbor)
			if new_neighbor_cost < self.global_best_X[1]:
				self.global_best_X = (iteration, new_neighbor_cost, new_neighbor.copy())
			
			# add to tabu_list_short
			old_recipes = np.setdiff1d(self._X[1], new_neighbor)
			tabu_list_short = np.append(tabu_list_short, old_recipes)
			# pop oldest from tabu_list_short
			while tabu_list_short.size > self.tabu_age[0]:
				tabu_list_short = np.delete(tabu_list_short, 0)

			# apply new neighbor as current solution
			self._X = (new_neighbor_cost, new_neighbor.copy())

			# add data to graph
			self.graph_data = np.append(self.graph_data, [[new_neighbor_cost], [self.global_best_X[1]]], 1)

			iteration += 1

		return self.global_best_X


	def load_data(self, filepath: Path) -> None:
		if not filepath.exists():
			raise ValueError('File doesn\'t exist')
		
		with filepath.open() as json_f:
			model_data = json.load(json_f)

			self._N_Rec = model_data['n']
			self._N_Ing = model_data['m']

			#TODO: add [today] to data_file (so that solution is only dependant on the data_file)
			self.today = 1
			#TODO: implement [money] restriction in neighborhood generation
			self.money = model_data['money']

			self._R = np.array(model_data['recipes'])
			self._T = np.array(model_data['times'])
			self._Q = np.array(model_data['quantities'])
			self._E = np.array(model_data['dates'])
			self._P = np.array(model_data['prices'])


	def set_params(self, alpha: float, beta: float, gamma:float) -> None:
		self.params = np.array([alpha, beta, gamma])


	def set_tabu_age(self, short: int, medium: int, long: int) -> None:
		self.tabu_age = np.array([short, medium, long])


	def generate_initial(self) -> None:
		new_X: np.ndarray = np.array([], dtype=int)
		for _ in range(self.recipe_count):
			curr_recipe = self.random_recipe_idx()
			while curr_recipe in new_X:
				curr_recipe = self.random_recipe_idx()
			new_X = np.append(new_X, [curr_recipe])
		self.initial_X = (self.calculate_cost_function(new_X), new_X)


	def generate_new(self, vec: np.ndarray, nbrhd_hamming: NeighborhoodType) -> np.ndarray:
		nbrhd_hamming = nbrhd_hamming.value
		new_vec: np.ndarray = vec.copy()
		aux_used_recipes: np.ndarray = np.array([])	# prevents duplication
		vec_idxs: np.ndarray = np.array([i for i in range(vec.size)])
		if nbrhd_hamming == 0:
			nbrhd_hamming = random.randrange(start=1, stop=7, step=1)
		if nbrhd_hamming > vec.size:
			nbrhd_hamming = vec.size
		
		for i in range(nbrhd_hamming):
			# pick idx in vec to change
			idx1 = random.choice(vec_idxs)
			vec_idxs = np.setdiff1d(vec_idxs, idx1)	# remove idx, so it cannot be changed again

			# add recipe from [idx1] to [aux_used_recipes]
			aux_used_recipes = np.append(aux_used_recipes, [vec[idx1]])

			# pick new recipe to place in [idx1]
			new_recipe: int = random.randrange(start=0, stop=self._N_Rec, step=1)
			if i == 0:	# ensure at least 1 changed recipe
				while new_recipe in vec:
					new_recipe = random.randrange(start=0, stop=self._N_Rec, step=1)
			else:
				while new_recipe in aux_used_recipes:
					new_recipe = random.randrange(start=0, stop=self._N_Rec, step=1)
			
			# mark [new_recipe] as already used, so it cannot duplicate
			aux_used_recipes = np.append(aux_used_recipes, [new_recipe])
			
			if new_recipe not in vec:
				new_vec[idx1] = new_recipe	# replace recipe in [idx1] with [new_recipe]

		return new_vec


	def generate_new_neighborhood(self, vec: np.ndarray, nbrhd_hamming: NeighborhoodType) -> np.ndarray:
		#TODO: maybe change to np.ndarray[np.ndarray]
		neighborhood: List[np.ndarray] = []
		for _ in range(self.neighborhood_size):
			# generete new neighbor that is not [already in list]/[== vec]
			new_neighbor: np.ndarray = self.generate_new(vec, nbrhd_hamming)
			while self.array_in_list(new_neighbor, neighborhood) or np.array_equal(new_neighbor, vec):
				new_neighbor = self.generate_new(vec, nbrhd_hamming)
			
			# add to neighborhood
			neighborhood.append(new_neighbor)

		return np.array(neighborhood)


	def calculate_cost_function(self, vec_X: np.ndarray, split_mode: bool = False) -> float:
		cost: np.ndarray = np.array([0.0, 0.0, 0.0]) # Shop, Loss, Time
		q_need: np.ndarray = np.zeros(self._N_Ing) # needed quantity of each Ingredient

		# j: recipe_idx
		for j in vec_X:
			cost[2] += self._T[j]	# add _R[j]'s time to Time-cost
			for i in range(self._N_Ing):
				q_need[i] += self._R[j, i]	# calc need for each Ingredient
		
		# calc Shop & Loss costs
		for i in range(self._N_Ing):
			#TODO: maybe write the [q_need[i]-self._Q[i]] to aux var and then determine to which cost to add
			cost[0] += self._P[i] * self.max(0, q_need[i] - self._Q[i])
			if self.determine_is_product_expired(i):
				cost[1] += self._P[i] * self.max(0, self._Q[i] - q_need[i])
		
		if split_mode:
			return cost
		else:
			return self.params @ cost


	def determine_is_product_expired(self, idx: int) -> bool:
		return self._E[idx] < self.today + 1


	def random_recipe_idx(self) -> int:
		# TODO: check out random.shuffle(x) -> shuffles list `x` randomly
		return random.randrange(start=0, stop=self._N_Rec, step=1)

	"""probably won't be used
	def random_bool(self) -> bool:
		rand_int = random.randrange(start=0, stop=100, step=1)
		return rand_int < 50 # true if less than 50, false if greater
	"""

	@staticmethod
	def max(a, b):
		if a > b:
			return a
		return b


	# check if [array] in [list of arrays]
	@staticmethod
	def array_in_list(array: np.ndarray, array_list: List):
		return next((True for elem in array_list if np.array_equal(elem, array)), False)
