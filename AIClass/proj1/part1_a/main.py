from kNN import kNN
import pandas as pd

df = pd.read_csv('AI/proj1/part1/data/labeled-examples', delimiter=' ', header=None)
df = df.rename(columns={0:'class', 1:'x', 2:'y', 3:'name'})

print(df.head(), '\n', df.shape)

#result = kNN.cross_validation(5, df)

kNN(df)

print()

while(not True):
    data_examples = {0: 'AI/proj1/part1/data/labeled-examples', 1: 'AI/proj1/part1/data/diabetes.csv', 2:'', 3:'Run all', 4: 'Enter Own'}
    selection = input(print(data_examples))
