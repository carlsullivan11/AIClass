import pandas as pd
import numpy as np

k = 5
n = 5

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
    
    for n in range(partition_num):
        start = round((n/partition_num) * df_len)
        stop = round(((n+1) / partition_num) * df_len)

        data_split.append(pd.DataFrame(df.iloc[start : stop])) # Split data in roughly equial parts
    
    for i in range(partition_num):
        df_train = (pd.merge(data_split[i], df, on=['class', 'x', 'y', 'name', 'prediction'], how='outer', indicator=True)
                            .query("_merge != 'both'")
                            .drop('_merge', axis=1)
                            .reset_index(drop=True))

        df_test = data_split[i]
        model_train.append(train_model(df_train))
        train_eval = eval(model_train[i], df_train)
        test_eval = eval(model_train[i], df_test)

        train_acc[0] += train_eval[0]   #[all]
        train_acc[1] += train_eval[1]   #[error]

        test_acc[0] += test_eval[0]     #[all]
        test_acc[1] += test_eval[1]     #[error]

    results = ('Test_acc_all: ' , test_acc[0]/partition_num , '\n' ,
                'Train_acc_all: ' , train_acc[0]/partition_num , '\n' ,
                'Test_acc_error' , test_acc[1]/partition_num , '\n' ,
                'Train_acc_error' , train_acc[1]/partition_num)

    return  results 

def train_model(df):
    store_all = pd.DataFrame()
    store_errors = store_k_errors(df)

    for i in df.index.to_list():
        sorted_df = knn_model(k, i , df, None)

        if (sorted_df['class'].iloc[1 : k+1].to_list().count(0) > k//2) :
            store_all = store_all_fun(sorted_df.iloc[:1], 0, store_all)
            
            if(sorted_df['class'].iloc[0] != 0):
                store_errors = store_errors_fun(sorted_df.iloc[:1], 1, store_errors)

        else:
            store_all = store_all_fun(sorted_df.iloc[:1], 1, store_all)

            if(sorted_df['class'].iloc[0] != 1):
                store_errors = store_errors_fun(sorted_df.iloc[:1], 0, store_errors)

    print(store_all.shape)
    print(store_errors.shape)
    return [store_all, store_errors]

def eval(model, test):
    store_all_acc = 0
    store_error_acc = 0

    for i in test.index.to_list():
        all_df = knn_model(k, i, model[0], test)
        error_df = knn_model(k, i, model[1], test)

        if (all_df['class'].iloc[1 : k+1].to_list().count(0) > k//2 and all_df['class'].iloc[0] == 0 
        or all_df['class'].iloc[1 : k+1].to_list().count(1) > k//2 and all_df['class'].iloc[0] == 1):
            store_all_acc += 1

        if (error_df['class'].iloc[1 : k+1].to_list().count(0) > k//2 and error_df['class'].iloc[0] == 0 
        or error_df['class'].iloc[1 : k+1].to_list().count(1) > k//2 and error_df['class'].iloc[0] == 1):
            store_error_acc += 1

    return [store_all_acc/test.shape[0], store_error_acc/test.shape[0]]

def knn_model(k, i, train, test):
        
        if(type(test) == type(None)):
            train['dist'] = eucl_dist(train.at[i,'x'], train.at[i,'y'], train['x'], train['y'])
            sorted_df = train.sort_values(by=['dist'])
            sorted_df = sorted_df.iloc[:k + 1]
        
        else:
            test['dist'] = eucl_dist(test.at[i,'x'], test.at[i,'y'], train['x'], train['y'])
            sorted_df = test.sort_values(by=['dist'])
            sorted_df = sorted_df.iloc[:k]

        return sorted_df[:k]

def store_k_errors(df):
    return pd.concat([df.sort_values('class').head(k), df.sort_values('class').tail(k)])
        
def eucl_dist(x_cord, y_cord, x_point, y_point):
        return (((x_cord - x_point)**2) + ((y_cord - y_point)**2))**(0.5)

def store_all_fun(df, value, store_all):
    index = df.head(1).index
    df.at[index[0] , 'prediction'] = value
    
    if (len(store_all.index) == 0):
        store_all = df
    else:
        store_all.loc[len(store_all.index)] = df.iloc[0]

    return store_all

def store_errors_fun(df, value, store_errors):
    index = df.head(1).index
    df.at[index[0] , 'prediction'] = value
    
    return pd.concat([store_errors, df]).reset_index(drop=True)
    