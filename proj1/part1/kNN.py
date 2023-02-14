import pandas as pd
import numpy as np

k = 3
n = 6

class kNN:
    def __init__(self, df):
        print(cross_validation(n , df))
        return None

def cross_validation(partition_num, df):
    df['prediction'] = np.nan
    data_split = []
    df_len = df.shape[0]
    test_acc = [0,0]        #[all, error]
    train_acc = [0,0]       #[all, error]
    model_train = []
    
    for i in range(partition_num):
        start = round((i/partition_num) * df_len)
        stop = round(((i+1) / partition_num) * df_len)

        data_split.append(pd.DataFrame(df.iloc[start : stop])) # Split data in roughly equial parts
  
        df_train = (pd.merge(data_split[i], df, on=['class', 'x', 'y', 'name', 'prediction'], how='outer', indicator=True)
                            .query("_merge != 'both'")
                            .drop('_merge', axis=1)
                            .reset_index(drop=True))

        df_test = data_split[i]
        model_train.append(train_model(df_train))

        train_eval = eval(model_train[i], df_train)
        test_eval = eval(model_train[i], df_test)

        writeFiles(i, 'StoreAll', model_train[i][0])
        writeFiles(i, 'StoreError', model_train[i][1])
        writeFiles(i, 'TrainDF', df_train)
        writeFiles(i, 'TestDF', df_test)

        train_acc[0] += train_eval[0]   #[all]
        train_acc[1] += train_eval[1]   #[error]

        test_acc[0] += test_eval[0]     #[all]
        test_acc[1] += test_eval[1]     #[error]

    results = ('Test_acc_all: ' , test_acc[0]/partition_num , '\n' ,
                'Train_acc_all: ' , train_acc[0]/partition_num , '\n' ,
                'Test_acc_error' , test_acc[1]/partition_num , '\n' ,
                'Train_acc_error' , train_acc[1]/partition_num)

    return results 

def train_model(df):
    store_all = pd.DataFrame(df)
    store_errors = store_k_errors(df)

    for i in df.index.to_list():
        sorted_df = knn_model(df, df.at[i,'x'], df.at[i,'y'])

        if (sorted_df['class'].iloc[:k].to_list().count(0) > k//2):
            store_all.at[i, 'prediction'] = 0
            if(sorted_df['class'].iloc[0] != 0):
                store_errors = store_errors_fun(df.iloc[[i]], 1, store_errors)
                #print('error - 0: ', df.at[i, 'class'])
        else:
            store_all.at[i, 'prediction'] = 1
            if(sorted_df['class'].iloc[0] != 1):
                store_errors = store_errors_fun(df.iloc[[i]], 0, store_errors)
                #print('error - 1: ', df.at[i, 'class'])
    print(store_errors.shape)
    return [store_all, store_errors]

def eval(model, test):
    store_all_acc = 0
    store_error_acc = 0
    
    
    for i in test.index.to_list():
        all_df = knn_model(model[0], test.at[i,'x'], test.at[i, 'y'])
        error_df = knn_model(model[1], test.at[i,'x'], test.at[i, 'y'])
        
        if (all_df['class'].iloc[:k].to_list().count(0) > k//2 and test.at[i, 'class'] == 0 
        or all_df['class'].iloc[:k].to_list().count(1) > k//2 and test.at[i, 'class'] == 1):
            store_all_acc += 1

        if (error_df['class'].iloc[:k].to_list().count(0) > k//2 and test.at[i, 'class'] == 0 
        or error_df['class'].iloc[:k].to_list().count(1) > k//2 and test.at[i, 'class'] == 1):
            store_error_acc += 1

    return [store_all_acc/test.shape[0], store_error_acc/test.shape[0]]

def knn_model(train, x, y): 
    train['dist'] = eucl_dist(np.array(train['x']), np.array(train['y']), x, y)
    sorted_df = train.nsmallest(k, 'dist')
    return sorted_df[:]

def store_k_errors(df):
    return pd.concat([df.sort_values('class').head(k), df.sort_values('class').tail(k)], ignore_index=True)
        
def eucl_dist(x_cord, y_cord, x_point, y_point):
    return np.array((((x_cord - x_point)**2) + ((y_cord - y_point)**2))**(0.5))

def store_errors_fun(df, value, store_errors):
    df.at[df.head(1).index[0], 'prediction'] = value
    return pd.concat([store_errors, df], ignore_index=True)

def writeFiles(i, type ,file):
    file = pd.DataFrame(file)
    file.to_csv('AI/proj1/part1/DataCSV/' + type + str(i) + '.csv', index=False)


def store_all_fun(df, value, store_all):
    df.at[df.head(1).index[0], 'prediction'] = value

    if (len(store_all.index) == 0):
        store_all = df
    else:
        store_all.loc[len(store_all.index)] = df.loc[0]

    return store_all    