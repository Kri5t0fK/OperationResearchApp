"""Operational Research App Model module

Enums:
	- NeighborhoodType
	- SolutionSelectionMethod
	
Classes:
	- Model

"""
from pathlib import Path 		# universal Path
from enum import Enum, unique 	# helpful
from typing import Tuple, List  # code visibility
import random 					# generating initial solutions
import json 					# data parsing

import numpy as np				# faster matrix math & utility

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
			return f'Ham{self.value}'
		else:
			return 'None'


@unique
class SolutionSelectionMethod(Enum):

	BEST = 1
	RANDOM = 2

	def __str__(self) -> str:
		if self.value == self.BEST.value:
			return 'Best'
		elif self.value == self.RANDOM.value:
			return 'Random'
		else:
			return 'None'


class Model:
	"""Model
	Contains all model data & logic for Taboo Search optimization

	"""
	
	def __init__(self):
		"""Initializes the model with zero values
		
		"""
		
		self.today: int = 0
		self.money: int = 0

		# matrices
		self._R: np.ndarray = np.array([])					# recipes (array of ingredient arrays)

		# vectors
		self._T: np.ndarray = np.array([])					# times
		self._Q: np.ndarray = np.array([])					# quantities
		self._E: np.ndarray = np.array([])					# expiry dates
		self._P: np.ndarray = np.array([])					# prices
		self._X: Tuple[float, np.ndarray]					# solution

		self._N_Rec: int = 0								# N_Rec -> number of recipies
		self._N_Ing: int = 0								# N_Ing -> number of ingredients

		# algorythm parameters
		self.recipe_count: int								# target recipe count (length of _X)
		self.initial_X: Tuple[float, np.ndarray]			# [cost_function_val, _X]
		self.global_best_X: Tuple[int, float, np.ndarray]	# [iteration, cost_function_val, _X]
		self.iteration_limit: int
		self.cutoff: int									# stop when under that value
		self.neighborhood_size: int
		self.aspiration_coeff: float
		self.tabu_age: np.ndarray = np.array([0, 0, 0])		# [short, medium, long]

		# cost function parameters
		self.params: np.ndarray = np.array([0.0, 0.0, 0.0])	# [alpha, beta, gamma]

		# data for gui graph of cost_function
		self.graph_data: np.ndarray[np.ndarray, np.ndarray]	# [current_cost, best_cost]

		random.seed() # seed the random number generator with system time (pretty random, random enough)

		
	def tabu_search(self, max_iterations: int, nbrhd_hamming: NeighborhoodType, \
				ssm_type: SolutionSelectionMethod) -> int:
		"""Taboo Search implementation for Model
		
		This implementation 

		Keyword arguments:
		argument -- description
		Return: return_description
		"""
		
		tabu_list_short 	= np.array([], dtype=int)
		self.global_best_X  = (0, self.initial_X[0], self.initial_X[1].copy())		# best solution
		self._X 			= (self.initial_X[0], self.initial_X[1].copy())			# current solution
		
		self.graph_data 	= np.array([[self.initial_X[0]], [self.initial_X[0]]])	# initialize graph data

		iteration: int = 1
		while iteration < max_iterations:
			neighborhood: np.ndarray = self.generate_new_neighborhood(self._X[1], nbrhd_hamming)
			neighborhood_non_taboo: np.ndarray = np.array([[]], dtype=int)	# pick only neighbors not forbidden by Tabu or meeting aspiration

			for candidate in neighborhood:
				non_taboo: bool = True

				for recipe in candidate:
					if recipe in tabu_list_short:
						non_taboo = False

				
				if not non_taboo:	# aspiration check
					if self.calculate_cost_function(candidate) < self.aspiration_coeff * self.global_best_X[1]: 
						non_taboo = True
				
				if non_taboo:
					if neighborhood_non_taboo.size == 0:
						neighborhood_non_taboo = np.append(neighborhood_non_taboo, [candidate], 1)
					else:
						neighborhood_non_taboo = np.append(neighborhood_non_taboo, [candidate], 0)
			
			
			if neighborhood_non_taboo.size == 0: #TODO: what if neighborhood_non_taboo empty? pick from [tabu_medium]? 
				neighborhood_non_taboo = np.append(neighborhood_non_taboo, [neighborhood[0]], 1) # force tabu solution

			new_neighbor: np.ndarray = neighborhood_non_taboo[0]	# select new neighbor

			if ssm_type == SolutionSelectionMethod.RANDOM:
				new_neighbor = random.choice(neighborhood_non_taboo)

			elif ssm_type == SolutionSelectionMethod.BEST:
				new_neighbor_cost: float = self.calculate_cost_function(new_neighbor)

				for candidate in neighborhood_non_taboo:

					candidate_cost: float = self.calculate_cost_function(candidate)
					
					if candidate_cost < new_neighbor_cost:
						new_neighbor = candidate
						new_neighbor_cost = candidate_cost
			
			
			new_neighbor_cost: float = self.calculate_cost_function(new_neighbor)

			if new_neighbor_cost < self.global_best_X[1]:					# check if new best
				self.global_best_X = (iteration, new_neighbor_cost, new_neighbor.copy())
			
			old_recipes = np.setdiff1d(self._X[1], new_neighbor)
			tabu_list_short = np.append(tabu_list_short, old_recipes)		# add to tabu_list_short
			
			while tabu_list_short.size > self.tabu_age[0]:					# pop oldest from tabu_list_short
				tabu_list_short = np.delete(tabu_list_short, 0)

			self._X = (new_neighbor_cost, new_neighbor.copy())				# apply new neighbor as current solution

			# update graph_data
			self.graph_data = np.append(self.graph_data, [[new_neighbor_cost], [self.global_best_X[1]]], 1)

			if new_neighbor_cost < self.cutoff:		# cutoff stop condition
				break

			iteration += 1

		return iteration


	def load_data(self, filepath: Path) -> None:
		"""Loads data from a .json file
		Assumes a correct file format (.json, as generated with /data_gen.py)
		"""

		if not filepath.exists():
			raise ValueError('File doesn\'t exist')
		
		with filepath.open() as json_f:
			model_data = json.load(json_f)

			self._N_Rec = model_data['n']
			self._N_Ing = model_data['m']

			#TODO: add [today] to data_file (so that solution is only dependant on the data_file)
			self.today = 1
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
		aux_used_recipes = np.array([], dtype=int)	# prevents duplication
		vec_idxs = np.array([i for i in range(vec.size)], dtype=int)

		if nbrhd_hamming == 0:
			nbrhd_hamming = random.randrange(start=1, stop=7, step=1)

		if nbrhd_hamming > vec.size:
			nbrhd_hamming = vec.size
		
		for i in range(nbrhd_hamming):
			
			idx1 = random.choice(vec_idxs)			# pick idx in vec to change
			vec_idxs = np.setdiff1d(vec_idxs, idx1)	# remove idx, so it cannot be changed again

			aux_used_recipes = np.append(aux_used_recipes, [vec[idx1]]) # add recipe from [idx1] to [aux_used_recipes]

			new_recipe: int = random.randrange(start=0, stop=self._N_Rec, step=1) # pick new recipe to place in [idx1]

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
		neighborhood: List[np.ndarray] = []

		for _ in range(self.neighborhood_size):
			new_neighbor: np.ndarray = self.generate_new(vec, nbrhd_hamming)		# generete new neighbor that is not [already in list]/[== vec]

			while self.is_array_in_list(new_neighbor, neighborhood) or np.array_equal(new_neighbor, vec):
				new_neighbor = self.generate_new(vec, nbrhd_hamming)
			
			neighborhood.append(new_neighbor)	# add to neighborhood

		return np.array(neighborhood)


	def calculate_cost_function(self, vec_X: np.ndarray, split_mode: bool = False) -> float:
		cost = np.array([0.0, 0.0, 0.0]) # Shop, Loss, Time
		q_need = np.zeros(self._N_Ing) # needed quantity of each Ingredient
		
		for j in vec_X:	# j: recipe_idx
			cost[2] += self._T[j]			# add _R[j]'s time to Time-cost

			for i in range(self._N_Ing):
				q_need[i] += self._R[j, i]	# calc need for each Ingredient
		
		for i in range(self._N_Ing): # calc Shop & Loss costs
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
		return random.randrange(start=0, stop=self._N_Rec, step=1)


	@staticmethod
	def max(a, b):
		if a > b:
			return a
		return b


	@staticmethod
	def is_array_in_list(array: np.ndarray, array_list: List):
		return next((True for elem in array_list if np.array_equal(elem, array)), False)
