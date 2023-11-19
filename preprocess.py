import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def preprocess_stage_1(path = '../data/training_data.csv'):
    df = pd.read_csv(path)
    # Getting City Names from address
    df['ADDRESS'] = df['ADDRESS'].apply(lambda x: x.split(',')[-1])

    kmeans = KMeans(n_clusters=20, random_state=0, n_init='auto').fit(df[['LATITUDE', 'LONGITUDE']])
    df['GRID_NUM'] = kmeans.labels_


    df['LOG_SQFT'] = np.log(df['SQUARE_FT'])
    # standard scale both sqft and price
    scaler = StandardScaler()
    df['LOG_SQFT'] = scaler.fit_transform(df[['LOG_SQFT']])
    df['SQUARE_FT'] = scaler.fit_transform(df[['SQUARE_FT']])
    df['LATITUDE'] = scaler.fit_transform(df[['LATITUDE']])
    df['LONGITUDE'] = scaler.fit_transform(df[['LONGITUDE']])
    return df

def get_data(path = '../data', do_pca = False):
    train_df = preprocess_stage_1(os.path.join(path, 'training_data.csv'))
    test_df = preprocess_stage_1(os.path.join(path, 'test_data.csv'))

    # one hot encode address for both train and test
    train_df = pd.get_dummies(train_df, columns=['ADDRESS'], drop_first=False)
    test_df = pd.get_dummies(test_df, columns=['ADDRESS'], drop_first=False)

    # drop the addresses that are not common in both train and test
    train_df = train_df.drop(columns=[col for col in train_df.columns if col not in test_df.columns])
    test_df = test_df.drop(columns=[col for col in test_df.columns if col not in train_df.columns])

    # one hot encode grid num and bhk_no for both train and test
    train_df = pd.get_dummies(train_df, columns=['GRID_NUM', 'BHK_NO.'], drop_first=False)
    test_df = pd.get_dummies(test_df, columns=['GRID_NUM', 'BHK_NO.'], drop_first=False)

    # drop the grid num and bhk_no that are not common in both train and test
    train_df = train_df.drop(columns=[col for col in train_df.columns if col not in test_df.columns])
    test_df = test_df.drop(columns=[col for col in test_df.columns if col not in train_df.columns])

    if do_pca:
        # now do pca on train and test
        pca = PCA(n_components=5)
        # do pca only on one hot encoded columns
        # get all columns which are not in NORMAL_COLS
        NORMAL_COLS = ['UNDER_CONSTRUCTION', 'RERA', 'BHK_NO.', 'SQUARE_FT', 'READY_TO_MOVE', 'RESALE', 'LATITUDE', 'LONGITUDE', 'LOG_SQFT', "GRID_NUM"]
        ABNORMAL_COLS = [col for col in list(train_df.columns) if col not in NORMAL_COLS]
        train_pca = pca.fit_transform(train_df[ABNORMAL_COLS])
        test_pca = pca.transform(test_df[ABNORMAL_COLS])

        prop_var = pca.explained_variance_ratio_
        eigenvalues = pca.explained_variance_
        PC_numbers = np.arange(pca.n_components_) + 1

        
        # add pca columns to train and test
        train_df['PCA_1'] = train_pca[:, 0]
        train_df['PCA_2'] = train_pca[:, 1]
        train_df['PCA_3'] = train_pca[:, 2]
        train_df['PCA_4'] = train_pca[:, 3]
        test_df['PCA_1'] = test_pca[:, 0]
        test_df['PCA_2'] = test_pca[:, 1]
        test_df['PCA_3'] = test_pca[:, 2]
        test_df['PCA_4'] = test_pca[:, 3]

        # drop the one hot encoded columns
        train_df = train_df.drop(columns=ABNORMAL_COLS)
        test_df = test_df.drop(columns=ABNORMAL_COLS)

    # convert all boolean columns to int
    for col in train_df.columns:
        if train_df[col].dtype == bool:
            train_df[col] = train_df[col].astype(np.int8)
            test_df[col] = test_df[col].astype(np.int8)
    
    # convert all object columns to int
    for col in train_df.columns:
        if train_df[col].dtype == object:
            train_df[col] = train_df[col].astype(np.int8)
            test_df[col] = test_df[col].astype(np.int8)
    
    # repeat the same for test
    for col in test_df.columns:
        if test_df[col].dtype == bool:
            train_df[col] = train_df[col].astype(np.int8)
            test_df[col] = test_df[col].astype(np.int8)
    
    # convert all object columns to int
    for col in test_df.columns:
        if test_df[col].dtype == object:
            train_df[col] = train_df[col].astype(np.int)
            test_df[col] = test_df[col].astype(np.int)
            
    return train_df, test_df


if __name__ == '__main__':
    train_df, test_df = get_data(do_pca = True)
    np.save('../data/ptrain_pca_4.npy', train_df)
    np.save('../data/ptest_pca_4.npy', test_df)
