import os
import pandas as pd
from sklearn.model_selection import train_test_split


def load_dataset(path):
    df = pd.read_csv(os.path.join(path,"train.csv"))
    return df

def split_data(df,ts):
    y = df['Transported']
    X = df.drop(columns='Transported')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=ts, random_state=42)
    
    
    return X_train,X_test,y_train,y_test

def save_dataset(X_train,X_test,y_train,y_test,path):
    os.makedirs(path,exist_ok=True)

    X_train.to_parquet(os.path.join(path,'X_train.parquet'),index=False)
    X_test.to_parquet(os.path.join(path,'X_test.parquet'),index=False)
    pd.DataFrame(y_train).to_parquet(os.path.join(path,'y_train.parquet'),index=False)
    pd.DataFrame(y_test).to_parquet(os.path.join(path,'y_test.parquet'),index=False)

    print('Saved Succesfully')


if __name__ == '__main__':
    read_path = 'data/raw/'
    write_path = 'data/processed/'
    raw_data = load_dataset(read_path)
    
    X_train,X_test,y_train,y_test = split_data(raw_data,0.33)
    save_dataset(X_train,X_test,y_train,y_test,write_path)
