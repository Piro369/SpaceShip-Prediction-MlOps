import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_dataset(path):
    df = pd.read_csv(path+"train.csv")
    return df

def split_data(df,ts):
    y = df['Transported']
    X = df.iloc[:,:-1]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=ts, random_state=42)
    
    
    return X_train,X_test,y_train,y_test

def save_dataset(X_train,X_test,y_train,y_test,path):
    X_train.to_csv(path+'X_train.csv')
    X_test.to_csv(path+'X_test.csv')
    y_train.to_csv(path+'y_train.csv')
    y_test.to_csv(path+'y_test.csv')

    print('Saved Succesfully')


if __name__ == '__main__':
    read_path = '../data/raw/'
    write_path = '../data/processed/'
    df = load_dataset(read_path)
    
    X_train,X_test,y_train,y_test = split_data(df,0.33)
    save_dataset( X_train,X_test,y_train,y_test,write_path )
