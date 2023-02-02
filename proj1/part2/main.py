import math
import ga_eval
import ga_util
import langermann_params
import shekel_params
import numpy as np


def simulated_annealing(problem, schedule, x):
    current = problem(x)
    t = 1
    print(current)
    while True:
        temp = schedule(t)
        t += 1
        if(temp == 0):
            return current
        next = np.random.shuffle(x)
        delta_e = current - problem(next)
        print(delta_e)
        if delta_e > 0:
            current = next
        else:
            probability = math.exp(delta_e/T)
            if probability > np.random.random():
                current = next

print(simulated_annealing(ga_eval.sphere, ga_util.schedule, np.ndarray(shape=(5,5))))
print(simulated_annealing(ga_eval.griew, ga_eval.griew_c))


def genetic_algorithm(population, fitness, condition):
    while not condition:
        weights = [fitness(ind) for ind in population]
        pop2 = []
        for i in range(len(population)):
            parent1, parent2 = np.random.random_sample(population, weights=weights, k=2)
            child = reproduce(parent1, parent2)
            if np.random.random() < 0.05:
                child = mutate(child)
            pop2.append(child)

        population = pop2

    return max(population, key=fitness)

def reproduce(parent1, parent2):
    parent1 = ga_util.binary(parent1)
    parent2 = ga_util.binary(parent2)

    n = len(parent1)
    c = np.random.randint(1,n)
    return ga_util.bitstr2float(parent1[:c] + parent2[c:])

def mutate(child):
    index = np.random.randint(0, len(child) - 1)
    mutation = (np.random.random() - 0.5) * 0.2
    child[index] += mutation
    return child
