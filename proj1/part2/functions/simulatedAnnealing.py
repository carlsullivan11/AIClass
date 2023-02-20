import random
import math

class sa:
    pass

''' Define a simulated annealing algorithm function that takes in problem, 
parameters, the input vector, and scheduling rate.'''
def simulated_annealing(problem, param, input_vector, schedule):
    # Initialize the current state and the temperature
    current = problem(input_vector)
    t = 1

    # Run the simulated annealing algorithm
    while True:
        # Update the temperature and check if the algorithm has reached the minimum temperature
        temp = 1 - (1 / schedule)
        t += 1
        if temp == 0:
            return current
        
        # Generate a new candidate state
        while True:
            next = random.random() * input_vector
            if param(next):
                break

        # Calculate the change between the current and candidate state
        delta_e = current - problem(next)

        # Determine whether to accept or reject the candidate state based on the Metropolis criterion
        if delta_e > 0:
            current = problem(next)
        else:
            probability = math.exp(delta_e / temp)
            # Added Probability of the next solution replacing current solution
            if probability > random.random():
                current = problem(next)

