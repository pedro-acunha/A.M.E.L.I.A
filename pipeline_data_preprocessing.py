import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def create_colours(data, features):
    """Create dataframe with colour-colour data. Return an Array and list of colours.

    Args:
        data (DataFrame): Dataframe with photometric data
        features (list): List of features from photometric data
    """
    N = len(data)
    F = len(features)
    n=0
    for i in np.linspace(1,len(features),len(features),dtype=int):
    	n = n + (i-1)

    df_features = np.zeros((N, n))
    y=0
    lista=[]
    for z in np.linspace(0,F,F,dtype=int):
    	for x in np.linspace(1,F-1,F-1,dtype=int):
        	if z!=x and z<x:
        		df_features[:,y] = data[features[z]] - data[features[x]]
        		y+=1
        		lista += [features[z]+'-'+features[x]]
        	else:
           		pass
    df_colours = pd.DataFrame(df_features,columns = lista)
    return df_colours

def DataCollection(task_type, path_data_train1, path_data_train2, path_data_test, list_features, list_add_relation, target_variable1, target_variable2, target_variable3, imputation):
    """Collects data to be used in train and test dataframes. Return Train and Test dataset.

    Args:
        task_type (string): Task type, should be Supervised, Semi-Supervised, Unsupervised
        path_data_train1 (string): Path for the Type II Quasars data
        path_data_train2 (string): Path for the Galaxies data
        path_data_test (string): Path for the data for the semi-supervised classification task
        list_features (list): List of features names in the dataframes
        list_add_relation (list): Relation to include in DataFrame
        target_variable1 (string): Target name for Type II Quasars
        target_variable2 (string): Target name for Galaxies
        target_variable3 (string): target Name for Test Data
        imputation (string): Check if imputation should be applied

    Returns:
        TYPE: Description
    """
    if task_type == 'Supervised':

        df_qso2_g = pd.read_csv(path_data_train1)
        df_qso2 = df_qso2_g[list_features].copy().drop_duplicates()
        df_qso2['class'] = target_variable1

        df_gal_g = pd.read_csv(path_data_train2)
        if len(df_gal_g)>len(df_qso2_g):
        	df_gal = df_gal_g[list_features].copy().drop_duplicates().sample(len(df_qso2_g),random_state=24)
        	df_gal['class'] = target_variable2
        else:
        	df_gal = df_gal_g[list_features].copy().drop_duplicates()
        	df_gal['class'] = target_variable2
            
        if imputation=='True':
            imputer = KNNImputer(missing_values=0, n_neighbors=6, weights='distance')
            df_qso2 = pd.DataFrame(imputer.fit_transform(df_qso2), columns=[list_features])
            imputer2 = KNNImputer(missing_values=9999, n_neighbors=6, weights='distance')
            df_gal = pd.DataFrame(imputer2.fit_transform(df_gal), columns=[list_features])
        else:
            pass

        df = pd.concat([df_qso2,df_gal]).reset_index()
        
        list_col = list_features
        # list_col=[]
        # check = ['modelMag_u', 'modelMag_g','modelMag_r','modelMag_i','modelMag_z']
        # if any(n in check for n in list_features):
        #     list_col = [e for e in list_features if e in check]

        df_colours = create_colours(df,list_col)
        df_train = pd.concat([df,df_colours.reindex(df.index)], axis=1, sort=False)

        if len(list_add_relation) != 0:
            list_relation = []
            for i in range(0,len(list_add_relation)):
                list_relation = list_add_relation[i].split('-')
                df_train[list_add_relation[i]] = df_train[list_relation[0]] - df_train[list_relation[1]]
                list_relation = []

        df_train['Target'] = pd.Series(np.where(df_train['class'].values == target_variable1, 1, 0),df_train.index)

        if 'index' in list(df_train.columns):
        	df_train = df_train.drop('index',axis=1)
        else:
        	pass

        features_list = df_train.columns.values.tolist()
        features_list.remove('Target')

        targets= ['Target']
        if 'class' in features_list:
        	features_list.remove('class')

        x_train, x_test, y_train, y_test = train_test_split(df_train[features_list], df_train[targets], test_size=0.2, shuffle =True, random_state=0)

        return x_train, x_test, y_train, y_test, features_list

    elif task_type == 'Semi-Supervised':
        df_qso2_g = pd.read_csv(path_data_train1)
        df_qso2 = df_qso2_g[list_features].copy().drop_duplicates()
        df_qso2['class'] = target_variable1

        df_gal_g = pd.read_csv(path_data_train2)
        if len(df_gal_g)>len(df_qso2_g):
        	df_gal = df_gal_g[list_features].copy().drop_duplicates().sample(len(df_qso2_g,random_state=24))
        	df_gal['class'] = target_variable2
        else:
        	df_gal = df_gal_g[list_features].copy().drop_duplicates()
        	df_gal['class'] = target_variable2

        if imputation=='True':
            imputer = KNNImputer(missing_values=0, n_neighbors=6, weights='distance')
            df_qso2 = pd.DataFrame(imputer.fit_transform(df_qso2), columns=[list_features])
            imputer2 = KNNImputer(missing_values=9999, n_neighbors=6, weights='distance')
            df_gal = pd.DataFrame(imputer2.fit_transform(df_gal), columns=[list_features])
        else:
            pass
        
        df = pd.concat([df_gal,df_qso2]).reset_index()

        df_test_g = pd.read_csv(path_data_test)
        df_test = df_test_g[list_features].copy()
        df_test['class'] = target_variable3

        list_col = list_features
        # list_col=[]
        # check = ['w1mpro','w2mpro','w3mpro','w4mpro']
        # if any(n in check for n in list_features):
        #     list_col = [e for e in list_features if e not in ('w1mpro','w2mpro','w3mpro','w4mpro')]
        # else:
        #     list_col = list_features

        df_train_colours = create_colours(df,list_col)
        df_test_colours = create_colours(df_test,list_col)

        df_train = pd.concat([df,df_train_colours.reindex(df.index)], axis=1, sort=False)
        df_test = pd.concat([df_test,df_test_colours.reindex(df_test.index)], axis=1, sort=False)

        if len(list_add_relation) != 0:
            list_relation = []
            for i in range(0,len(list_add_relation)):
                list_relation = list_add_relation[i].split('-')
                df_train[list_add_relation[i]] = df_train[list_relation[0]] - df_train[list_relation[1]]
                df_test[list_add_relation[i]] = df_test[list_relation[0]] - df_test[list_relation[1]]
                list_relation = []

        df_train['Target'] = pd.Series(np.where(df_train['class'].values == target_variable1, 1, 0),df_train.index)
        df_test['Target'] = pd.Series(np.where(df_test['class'].values == target_variable1, 1, 0),df_test.index)

        if 'index' in list(df_train.columns):
        	df_train = df_train.drop('index',axis=1)

        features_list = df_train.columns.values.tolist()
        features_list.remove('Target')

        if 'class' in features_list:
            features_list.remove('class')

        x_train = df_train.drop(['class','Target'],1)
        x_test = df_test.drop(['class','Target'],1)

        y_train = pd.DataFrame(df_train['Target'], columns=['Target'])
        y_test = pd.DataFrame(df_test['Target'], columns=['Target'])

    elif task_type == 'Unsupervised':
        df_qso2_g = pd.read_csv(path_data_train1)
        df_qso2 = df_qso2_g[list_features].copy().drop_duplicates()

        df_gal_g = pd.read_csv(path_data_train2)
        df_gal = df_gal_g[list_features].copy().drop_duplicates()
        df = pd.concat([df_gal,df_qso2]).reset_index()
        
        if imputation=='True':
            imputer = KNNImputer(missing_values=0, n_neighbors=6, weights='distance')
            df_qso2 = pd.DataFrame(imputer.fit_transform(df_qso2), columns=[list_features])
            imputer2 = KNNImputer(missing_values=9999, n_neighbors=6, weights='distance')
            df_gal = pd.DataFrame(imputer2.fit_transform(df_gal), columns=[list_features])
        else:
            pass

        if len(list_add_relation) != 0:
            list_relation = []
            for i in range(0,len(list_add_relation)):
                list_relation = list_add_relation[i].split('-')
                df[list_add_relation[i]] = df[list_relation[0]] - df[list_relation[1]]
                list_relation = []

    else:
        pass

    if task_type == 'Supervised' or task_type == 'Semi-Supervised':
        return x_train, x_test, y_train, y_test, features_list
    elif task_type == 'Unsupervised':
        return df
    else:
    	print("Are you sure about that task_type?")



def DataPreprocessing(scaling_method, dataframe_Xtrain, dataframe_Xtest, dataframe_ytrain, dataframe_ytest, features):
    """Removes duplicates, check for outliers and scales the data

    Args:
        scaling_method (string): Scaling method, needs to be Robust or Min-Max
        dataframe_Xtrain (DataFrame): Training Features database
        dataframe_Xtest (DataFrame): Testing Features database
        dataframe_ytrain (DataFrame): Training Target database
        dataframe_ytest (DataFrame): Testing Target database
        features (list): List of features

    Returns:
        TYPE: Description

    """
    X_train = dataframe_Xtrain
    X_test = dataframe_Xtest
    Y_train = dataframe_ytrain
    Y_test = dataframe_ytest

    if scaling_method == 'Robust':
        scaler = RobustScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_train = pd.DataFrame(X_train, columns=features)
        X_test = scaler.transform(X_test)
        X_test = pd.DataFrame(X_test, columns=features)

    elif scaling_method == 'Min-Max':
        scaler = MinMaxScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_train = pd.DataFrame(X_train, columns=features)
        X_test = scaler.transform(X_test)
        X_test = pd.DataFrame(X_test, columns=features)

    else:
        print('Please write Robust or Min-Max.')

    return X_train, X_test, Y_train, Y_test


def DataToClassifier(task_type, scaling_method, path_data_train1, path_data_train2, path_data_test, list_features, list_add_relation , target_variable1, target_variable2,target_variable3):
    """Global functions to extract and pre-process data for classification

    Args:
        path_data_train1 (string): Path for the Type II Quasars data
        path_data_train2 (string): Path for the Galaxies data
        path_data_test (string): Path for the data for the semi-supervised classification task
        list_features (list): List of features names in the dataframes
        target_variable1 (string): Target name for Type II Quasars
        target_variable2 (string): Target name for Galaxies
        targets (list): List with strings with targets names

    Returns:
        DataFrame: DataFrames with training and test data
    """
    x_train, x_test, y_train, y_test, features_list = DataCollection(task_type, path_data_train1, path_data_train2, path_data_test, list_features, list_add_relation, target_variable1, target_variable2, target_variable3)
    X_train, X_test, Y_train, Y_test = DataPreprocessing(scaling_method,x_train, x_test, y_train, y_test, features_list)

    return X_train, X_test, Y_train, Y_test
