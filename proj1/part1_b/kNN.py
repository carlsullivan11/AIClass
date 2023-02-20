import pandas as pd
import numpy as np

k = 3
n = 5

# kNN algorithm class
class kNN_commented:
    # initialize kNN class and execute cross validation
    def __init__(self, df):
        # call cross validation method
        cross_validation(n , df)
        return None

# partition data into n parts, train the model n times with different partition, and calculate accuracy of predictions
def cross_validation(partition_num, df):
    # add empty prediction column to data frame
    df['prediction'] = np.nan
    # list to store partitioned data frames
    data_split = []
    # number of rows in data frame
    df_len = df.shape[0]
    # list to store accuracy of model predictions for test data and train data
    test_acc = [0,0]        #[all, error]
    train_acc = [0,0]       #[all, error]
    # list to store trained models
    model_train = []
    
    # partition data into partition_num parts
    for n in range(partition_num):
        # determine starting and stopping indices for partition
        start = round((n/partition_num) * df_len)
        stop = round(((n+1) / partition_num) * df_len)

        # store partitioned data frame in data_split
        data_split.append(pd.DataFrame(df.iloc[start : stop])) # Split data in roughly equial parts
    
    # train model and evaluate accuracy for each partition of the data
    for i in range(partition_num):
        # separate partitioned data into training data (not the current partition) and test data (the current partition)
        df_train = (pd.merge(data_split[i], df, on=['class', 'x', 'y', 'name', 'prediction'], how='outer', indicator=True)
                            .query("_merge != 'both'")
                            .drop('_merge', axis=1)
                            .reset_index(drop=True))

        df_test = data_split[i]
        # train model on training data and evaluate accuracy on training and test data
        model_train = train_model(df_train)
        train_eval = eval(model_train, df_train)
        test_eval = eval(model_train, df_test)

        # store trained models and accuracy data in files for later use
        print(model_train[0].shape, ' : Store all')
        writeFiles(i, 'StoreAll', model_train[0])
        writeFiles(i, 'StoreError', model_train[1])
        writeFiles(i, 'TrainDF', df_train)
        writeFiles(i, 'TestDF', df_test)

        # sum accuracy data for each partition
        train_acc[0] += train_eval[0]   #[all]
        train_acc[1] += train_eval[1]   #[error]
        test_acc[0] += test_eval[0]     #[all]
        test_acc[1] += test_eval[1]     #[error]

    # calculate and print average accuracy data across partitions
    results = ('Test_acc_all: ' + str(round(test_acc[0]/partition_num, 2)*100) + '\n' +
                'Train_acc_all: ' + str(round(train_acc[0]/partition_num, 2)*100) + '\n' +
                'Test_acc_error: ' + str(round(test_acc[1]/partition_num, 2)*100) + '\n' +
                'Train_acc_error: ' + str(round(train_acc[1]/partition_num, 2)*100) + '\n')
    print(results)

def train_model(df):
    # create empty data frames to sstore model data and errors
    store_all = pd.DataFrame()
    store_errors = store_k_errors(df)
    # iterate over the index list of the input data
    for i in df.index.to_list():
        # use kNN model to get sorted data for each index in the data
        sorted_df = knn_model(k, i , df, None)

        # check if the majority of the k nearest neighbors belong to class 0
        if (sorted_df['class'].iloc[1 : k+1].to_list().count(0) > k//2) :
            # add the index data to the store all data frame if class is 0
            store_all = store_all_fun(sorted_df.iloc[:1], 0, store_all)
            
            # add the index data to the store errors data frame if class is not 0
            if(sorted_df['class'].iloc[0] != 0):
                store_errors = store_errors_fun(sorted_df.iloc[:1], 1, store_errors)

        # if majority of k nearest neighbors belong to class 1
        else:
            # add the index data to the store all data frame if class is 1
            store_all = store_all_fun(sorted_df.iloc[:1], 1, store_all)

            # add the index data to the store errors data frame if class is not 1
            if(sorted_df['class'].iloc[0] != 1):
                store_errors = store_errors_fun(sorted_df.iloc[:1], 0, store_errors)

    # print the shapes of the data frames and return them as tuple
    print(store_all.shape)
    print(store_errors.shape)
    return (store_all, store_errors)

    
def eval(model, test):
    
    # retrieve the two sets of data from the model output
    store_all = model[0]
    store_error = model[1]
    
    # initialize two variables to keep track of accuracy
    store_all_acc = 0
    store_error_acc = 0
    
    # iterate over each instance in the test data
    for i in test.index.to_list():
        
        # get the k nearest neighbors for the instance from both sets of stored data
        all_df = knn_model(k, i, store_all, test)
        error_df = knn_model(k, i, store_error, test)
        
        # check if the predicted class matches the true class based on the k nearest neighbors
        # if so, increment the respective accuracy counter
        if (all_df['class'].iloc[1 : k+1].to_list().count(0) > k//2 and all_df['class'].iloc[0] == 0 
        or all_df['class'].iloc[1 : k+1].to_list().count(1) > k//2 and all_df['class'].iloc[0] == 1):
            store_all_acc += 1

        if (error_df['class'].iloc[1 : k+1].to_list().count(0) > k//2 and error_df['class'].iloc[0] == 0 
        or error_df['class'].iloc[1 : k+1].to_list().count(1) > k//2 and error_df['class'].iloc[0] == 1):
            store_error_acc += 1
            
    # calculate the accuracy as a fraction of correctly classified instances over the total number of instances
    return [store_all_acc/test.shape[0], store_error_acc/test.shape[0]]

# this function takes in four parameters: k, index, trained data, and testing data.
def knn_model(k, i, train, test):
    
    # if the initial training(test data set to None), calculate the euclidean distance between training data and the given point i.
    if(type(test) == type(None)):
        train['dist'] = eucl_dist(train.at[i,'x'], train.at[i,'y'], train['x'], train['y'])
        # sort the dataframe based on the distances calculated, and return the top k + 1 points (since the first point is the query point itself).
        sorted_df = train.sort_values(by=['dist'])
        sorted_df = sorted_df.iloc[:k + 1]
    
    # otherwise, add the given point in the train dataframe, calculate the euclidean distance between trained data and the given point i.
    else:
        train.loc[len(train.index)] = test.loc[i]
        train['dist'] = eucl_dist(test.at[i,'x'], test.at[i,'y'], train['x'], train['y'])
        # Sort the dataframe based on the distances calculated, and return the top k points.
        sorted_df = train.sort_values(by=['dist'])
        sorted_df = sorted_df.iloc[:k]

    # return the sorted dataframe containing k nearest neighbors.
    return sorted_df[:k]

# this function returns the top and bottom k instances from a dataframe sorted by the 'class' column
def store_k_errors(df):
    return pd.concat([df.sort_values('class').head(k), df.sort_values('class').tail(k)])

# this function calculates the Euclidean distance between two points in a two-dimensional space
def eucl_dist(x_cord, y_cord, x_point, y_point):
    return (((x_cord - x_point)**2) + ((y_cord - y_point)**2))**(0.5)




def store_all_fun(df, value, store_all):
    # adds a prediction value to the 'prediction' column of the input dataframe
    index = df.head(1).index
    df.at[index[0], 'prediction'] = value
    # if the dataframe is empty, it sets the input dataframe to be the output dataframe
    if (len(store_all.index) == 0):
        store_all = df
    # otherwise, it appends the input dataframe to the output dataframe
    else:
        store_all.loc[len(store_all.index)] = df.iloc[0]

    return store_all

# this function returns the top and bottom k instances from a dataframe sorted by the 'class' column
def store_k_errors(df):
    return pd.concat([df.sort_values('class').head(k), df.sort_values('class').tail(k)])

# this function calculates the Euclidean distance between two points in a two-dimensional space
def eucl_dist(x_cord, y_cord, x_point, y_point):
    return (((x_cord - x_point)**2) + ((y_cord - y_point)**2))**(0.5)


def store_all_fun(df, value, store_all):
    # this function adds a prediction value to the 'prediction' column of the input dataframe
    index = df.head(1).index
    df.at[index[0], 'prediction'] = value

    # if the dataframe is empty, it sets the input dataframe to be the output dataframe    
    if (len(store_all.index) == 0):
        store_all = df
    # otherwise, it appends the input dataframe to the output dataframe
    else:
        store_all.loc[len(store_all.index)] = df.iloc[0]

    return store_all

def store_errors_fun(df, value, store_errors):
    # get the index of the first row of the dataframe
    index = df.head(1).index
    # set the value to the 'prediction' column of the row at the given index
    df.at[index[0] , 'prediction'] = value
    
    # concatenate the store_errors dataframe and the input dataframe, reset the index, and return the result
    return pd.concat([store_errors, df]).reset_index(drop=True)

def writeFiles(i, type ,file):
    # convert the input file list to a pandas dataframe
    file = pd.DataFrame(file)
    # write the dataframe to a CSV file with the given name in the specified directory
    file.to_csv('proj1/part1_b/DataCSV/' + type + str(i) + '.csv', index=False)