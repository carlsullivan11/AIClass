import math
import random
from functions import geneticAlgo
from functions import simulatedAnnealing
from functions import ga_eval
import numpy as np
import csv

eval_functions = [ga_eval.sphere, ga_eval.griew, ga_eval.langermann, ga_eval.shekel, ga_eval.micha]
eval_param = [ga_eval.sphere_c, ga_eval.griew_c, ga_eval.langermann_c, ga_eval.shekel_c, ga_eval.micha_c]

'''Driver for Simulated Annealing'''

file = open('proj1/part2/DataCSV/scheduling.csv', 'w', newline ='\n')
file_header = ['eval.problem', 'schedule', 'array', 'result1', 'result2', 'result3', 'result4', 'result5', 'average', 'min']

with file:
    write = csv.writer(file)
    write.writerow(file_header)
file.close()

eval_str = ['sphere', 'griew', 'langermann', 'shekel', 'micha']
input_array = [[-5,5], [0,200], [0,10], [0,10], [-100,100]]

j = 0
schedule = 100
for i in range(6):
    
    for ii in range(len(eval_functions)):
        sa_resutls = []
        low = input_array[ii][0]
        high = input_array[ii][1]
        first = random.randint(low, high)
        second = random.randint(low, high)

        for iii in range(5):
            sa_resutls.append(simulatedAnnealing.simulated_annealing(eval_functions[ii], eval_param[ii], np.array([first, second]), schedule))
            print(sa_resutls)

        sa_resutls.append(sum(sa_resutls)/len(sa_resutls))
        sa_resutls.append(min(sa_resutls))

        write_file = [eval_str[ii], schedule, '[' + str(first) + ',' + str(second) + ']']
        write_file.extend(sa_resutls)
        print(write_file)

        file = open('proj1/part2/DataCSV/scheduling.csv', 'a', newline ='\n')
        with file:
            write = csv.writer(file)
            write.writerow(write_file)

        j += 1
    schedule *= 10

file.close() 


'''Driver for Genetic Algorithm'''

file = open('proj1/part2/DataCSV/ga_results.csv', 'w', newline ='\n')
file_header = ['eval.problem', 'bit.len', 'pop.size', 'generations', 'mutation.rate', 'result1', 'result2', 'result3', 'result4', 'result5', 'average', 'min']
eval_str = ['sphere', 'griew', 'bump', 'langermann', 'shekel', 'odd_square']
eval_functions = [ga_eval.sphere, ga_eval.griew, ga_eval.langermann, ga_eval.shekel, ga_eval.odd_square]
with file:
    write = csv.writer(file)
    write.writerow(file_header)
file.close()

j = 0
for eval in eval_functions:
    pop_size = [5, 10, 15, 20]
    generations = [50, 75, 100, 150]
    mutation_rate = [0.025, 0.05, 0.075, 0.10]
    bit_len = 52
    

    for pop in pop_size:
        index_gen = 0
        index_mutation = 0

        for gen in generations:
            for mut in mutation_rate:
                ga_resutls = []
                for i in range(5):
                    ga_resutls.append(geneticAlgo.genetic_algorithm(pop, bit_len, gen, eval, mut))
                    print("Best: ", ga_resutls[i])

                ga_resutls.append(sum(ga_resutls)/len(ga_resutls))
                ga_resutls.append(min(ga_resutls))

                write_file = [eval_str[j], bit_len, pop, gen, mut]
                write_file.extend(ga_resutls)
                print(write_file)

                file = open('proj1/part2/DataCSV/ga_results.csv', 'a', newline ='\n')
                with file:
                    write = csv.writer(file)
                    write.writerow(write_file)
        

    j += 1

file.close()



    
