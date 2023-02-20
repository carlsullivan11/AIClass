from kNN import kNN
import pandas as pd

while(True):
    data_examples = {0: ['AI/AIClass/AIClass/proj1/part1_b/data/labeled-examples', ' ', None], 
                    1: ['AI/AIClass/AIClass/proj1/part1_b/data/diabetes.csv', ',', 0], 
                    2:['AI/AIClass/AIClass/proj1/part1_b/data/airlines_delay.csv', ',', 0], 
                    3:['Run all', ' '], 
                    4: ['Enter Own', ' ']}

    for x in data_examples.items():
        print(x[0], ':', x[1][0])
    selection = int(input('Input Number: '))
    
    
    if int(selection) in data_examples.keys():
        file = data_examples.get(selection)

        df = pd.read_csv(file[0], delimiter= file[1], header= file[2])
        print('Select columns (#) using following format \n',
        'Class:# - For Classification \n',
        'x:# - For x axis values\n',
        'y:# - For y axis values\n',
        'Name:# - For unique identifiers (If no ID then use same number as class)\n')

        if(file[2] == None):
            col_names = ['class', 'x', 'y', 'name']
            selection = []
            for name in col_names:
                selection.append([name , int(input(name + ':Column Number = '))])
        else:
            i = 0
            for col in df.columns:
                print('Column Index: ', i, '-->', col)
                i += 1
            print()
            col_names = ['class', 'x', 'y', 'name']
            selection = []
            for name in col_names:
                selection.append([name , int(input(name + ':Column Number = '))])
        
        new_df = pd.DataFrame()
        for select in selection:
            new_df[select[0]] = df.iloc[:,select[1]]

        kNN(new_df)

    else:
        print('Invalid selection, please select again')