import random
from time import time
from copy import deepcopy
from Queens import *
from tree_less_logic import *
import concurrent.futures
# Constants
POPULATION_SIZE = 100
GENERATIONS = 200
MUTATION_RATE = 0.1

# Parameter ranges
MAX_TRIES_RANGE = (10, 1000)
MAX_FLIPS_RANGE = (10, 1000)
TEMPERATURE_RANGE = (0.01, 1.0)
# The rule to satisfy
rule = "(AYB)O(C>D)"
def walk_sat(max_tries, max_flips, temperature):
    # Placeholder for the walkSat algorithm
    start_time = time()
    # Simulate some work
    w_obj = WalkSat(rule, max_flips, max_tries, temperature)
    sat, result = w_obj.SAT_till_SAT()
    # Assume it finds a solution
    return time() - start_time

class Agent:
    def __init__(self, max_tries, max_flips, temperature):
        self.max_tries = max_tries
        self.max_flips = max_flips
        self.temperature = temperature
        self.fitness = None

    @staticmethod
    def random_agent():
        return Agent(
            max_tries=random.randint(*MAX_TRIES_RANGE),
            max_flips=random.randint(*MAX_FLIPS_RANGE),
            temperature=random.uniform(*TEMPERATURE_RANGE)
        )

    def mutate(self):
        if random.random() < MUTATION_RATE:
            self.max_tries = random.randint(*MAX_TRIES_RANGE)
        if random.random() < MUTATION_RATE:
            self.max_flips = random.randint(*MAX_FLIPS_RANGE)
        if random.random() < MUTATION_RATE:
            self.temperature = random.uniform(*TEMPERATURE_RANGE)

def evaluate_fitness(agent):
    agent.fitness = walk_sat(agent.max_tries, agent.max_flips, agent.temperature)

def selection(population):
    # Select two agents with the lowest fitness (fastest)
    sorted_pop = sorted(population, key=lambda x: x.fitness)
    return sorted_pop[:2]

def crossover(parent1, parent2):
    child1 = deepcopy(parent1)
    child2 = deepcopy(parent2)
    # Single point crossover
    if random.random() > 0.5:
        child1.max_tries, child2.max_tries = child2.max_tries, child1.max_tries
    if random.random() > 0.5:
        child1.max_flips, child2.max_flips = child2.max_flips, child1.max_flips
    if random.random() > 0.5:
        child1.temperature, child2.temperature = child2.temperature, child1.temperature
    return child1, child2

def genetic_algorithm():
    # Initialize population
    population = [Agent.random_agent() for _ in range(POPULATION_SIZE)]

    for generation in range(GENERATIONS):
        # Evaluate fitness in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(evaluate_fitness, population)

        # Selection
        parents = selection(population)

        # Crossover
        children = []
        while len(children) < POPULATION_SIZE - len(parents):
            parent1, parent2 = random.sample(parents, 2)
            child1, child2 = crossover(parent1, parent2)
            children.extend([child1, child2])

        # Truncate excess children
        children = children[:POPULATION_SIZE - len(parents)]

        # Mutation
        for child in children:
            child.mutate()

        # Create new population
        population = parents + children

        # Logging
        best_fitness = min(agent.fitness for agent in population)
        print(f"Generation {generation + 1}: Best Fitness = {best_fitness:.4f}")

    # Best agent
    best_agent = min(population, key=lambda x: x.fitness)
    print("Best Agent:")
    print(f"Max Tries: {best_agent.max_tries}")
    print(f"Max Flips: {best_agent.max_flips}")
    print(f"Temperature: {best_agent.temperature:.4f}")
    print(f"Fitness (Time): {best_agent.fitness:.4f} seconds")
if __name__ == "__main__":
    genetic_algorithm()