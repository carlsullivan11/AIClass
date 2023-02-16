from kNN import kNN
import pandas as pd

df = pd.read_csv('proj1/part1/data/labeled-examples', delimiter=' ', header=None)

df = df.rename(columns={0:'class', 1:'x', 2:'y', 3:'name'})

print(df.head(), '\n', df.shape)

kNN(df)
