import random
import numpy as np
from functions import ga_util

class ga:
    pass

''' Define a genetic algorithm function that takes in population size, 
the length of bit strings, the number of generations to run, a fitness function, 
and a mutation probability.'''
def genetic_algorithm(pop_size, float_length, generations, fitness, mutation):
        
    # Generate an initial population of bit strings of length float_length
    population = generate_population(pop_size, float_length)
    
    # Create an empty list for the next generation of the population
    pop2 = []

    # Loop through the specified number of generations
    for i in range(generations):
        
        # Select two parents from the current population based on fitness
        parent = select_parents(population, fitness)
        
        # Produce a child by combining the two parents
        child = reproduce(parent[0], parent[1])
        
        # Mutate the child with a specified probability and add the result to the new population
        pop2.append(mutate(child[0], mutation))
        pop2.append(mutate(child[1], mutation))
        
        # Generate additional random individuals and add them to the new population
        pop2.extend(generate_population(pop_size - 2, float_length))
        
        # Replace the old population with the new one
        population = pop2

    # Evaluate the fitness of each individual in the final population
    fitness_values = [(fitness(np.array([ga_util.bitstr2float(ind)])), ind) for ind in population]
    
    # Sort the fitness values to find the individual with the highest fitness
    fitness_values = sorted(fitness_values, key=lambda l:l[0])
    best_fit = fitness_values[0][1]
    
    # Convert the best individual from a bit string to a float and return it
    return ga_util.bitstr2float(best_fit)

# Define a function to generate a population of bit strings of a specified length and size.
def generate_population(pop_size, length):
    # Initialize an empty list to store the population
    population = []
    
    # Loop through the specified population size and generate a random bit string of the given length for each individual
    for i in range(pop_size):
        float_bit_value = ''
        for ii in range(length):
            # Add a random 0 or 1 bit to the bit string
            float_bit_value += str(random.randint(0, 1))
        # Add the new bit string to the population
        population.append(float_bit_value)
    
    # Return the generated population
    return population

# Define a function to select two parents from a population based on fitness.
def select_parents(population, fitness):

    # Evaluate the fitness of each individual in the population
    fitness_values = [ (fitness(np.array([ga_util.bitstr2float(ind)])) , ind) for ind in population]
    
    # Sort the fitness values in ascending order
    fitness_values = sorted(fitness_values, key=lambda l:l[0])
    
    # Select the two individuals with the highest fitness as parents
    parents = [x[1] for x in fitness_values[:2]]
    
    # Return the selected parents
    return parents

# Define a function that takes two parent individuals and returns two child individuals by swapping genetic information between the parents at a randomly chosen crossover point.
def reproduce(parent1, parent2):
    # Determine the length of the parent bit strings
    n = len(parent1)
    
    # Choose a random crossover point
    c = np.random.randint(1,n)
    
    # Swap genetic information between the parents at the crossover point to produce two new child bit strings
    child1 = parent1[:c] + parent2[c:]
    child2 = parent2[:c] + parent1[c:]
    
    # Return the new child bit strings
    return [child1, child2]

# Define a function that takes a child individual and a mutation rate and returns the child with a bit mutation with a probability of mutation.
def mutate(child: str, mutation: float):
    # Generate a random number between 0 and 1, and check if it is less than the mutation rate
    if random.random() <= mutation:
        # If the random number is less than the mutation rate, select a random index in the child bit string to mutate
        new_child = ''
        index = random.randint(0, len(child))
        j = 0
        for i in child:
            # If the index is reached, flip the bit at that position
            if j == index:
                new_child += str(random.randint(0,1))
            else:
                new_child += i
            j += 1
        # Return the mutated child
        return new_child
    # If the random number is less than or equal to the mutation rate, return the original child
    else:
        return child