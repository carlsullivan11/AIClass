from kNN import kNN
import pandas as pd

df = pd.read_csv('/home/oghoodrattz/AI/proj1/data/labeled-examples', delimiter=' ', header=None)
df = df.rename(columns={0:'class', 1:'x', 2:'y', 3:'name'})

print(df.head(), '\n', df.shape)

#result = kNN.cross_validation(5, df)

kNN(df)

print()