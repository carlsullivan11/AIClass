import math
import random
import ga_eval
import ga_util
import numpy as np


def simulated_annealing(problem, param, x):
    current = problem(x)
    t = 1
    while True:
        temp = ga_util.schedule(t)
        t += 1
        if(temp == 0):
            return current
        
        while True:
            next = random.random() * x
            
            if(param(next)):
                break

        delta_e = current - problem(next)

        if delta_e > 0:
            current = problem(next)
        else:
            probability = math.exp(delta_e/temp)
            if probability > random.random():
                current = problem(next)



def genetic_algorithm(pop_size, float_length, generations, fitness):
        
    population = generate_population(pop_size, float_length)
    pop2 = []

    for i in range(generations):
        parent = select_parents(population, fitness)
        child = reproduce(parent[0], parent[1])
        pop2.append(mutate(child[0]))
        pop2.append(mutate(child[1]))
        pop2.extend(generate_population(pop_size - 2, float_length))
        population = pop2

    fitness_values = [(fitness(np.array([ga_util.bitstr2float(ind)])), ind) for ind in population]
    fitness_values.sort(key=lambda x: x[0])
    best_fit = fitness_values[0][1]
    return ga_util.bitstr2float(best_fit)

def generate_population(pop_size, length):
    population = []
    
    for i in range(pop_size):
        float_bit_value = ''
        for ii in range(length):
            float_bit_value += str(random.randint(0, 1))
        population.append(float_bit_value)
    return population

def select_parents(population, fitness):

    fitness_values = [(fitness(np.array([ga_util.bitstr2float(ind)])), ind) for ind in population]
    fitness_values.sort(key=lambda x: x[0])
    parents = [x[1] for x in fitness_values[:2]]
    return parents

def reproduce(parent1, parent2):
    n = len(parent1)
    c = np.random.randint(1,n)
    return [parent1[:c] + parent2[c:] , parent2[:c] + parent1[c:]]

def mutate(child : str):
    if random.random() < 0.05:
        new_child = ''
        index = random.randint(0, len(child))
        j = 0
        for i in child:
            if j == index:
                new_child += str(random.randint(0,1))
            else:
                new_child += i
            j += 1
        child = new_child
    return child




#print(simulated_annealing(ga_eval.sphere, ga_eval.sphere_c, np.array([5,5])))

#print(simulated_annealing(ga_eval.griew, ga_eval.griew_c, np.array([10,10])))

for i in range(5):
    print("Best: ", genetic_algorithm(20, 52, 100, ga_eval.shekel))
