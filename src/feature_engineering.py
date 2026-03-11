#1 load training and testing data
#2 scale the training  data
#3 save the scaled data into processed folder
import pickle
import pandas as pd
from data_preprocessing import load_and_split_data
from sklearn.preprocessing import StandardScaler

x_train,x_test,y_train,y_test=load_and_split_data()

scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.fit_transform(x_test)
pd.DataFrame(x_train_scaled).to_csv('../data/processed/x_train_scaled.csv',index=False)
pd.DataFrame(x_test_scaled).to_csv('../data/processed/x_test_scaled.csv',index=False)
pd.DataFrame(y_train).to_csv('../data/processed/y_train.csv',index=False)
pd.DataFrame(y_test).to_csv('../data/processed/y_test.csv',index=False)


with open('../artifacts/scaler.pkl','wb') as f:
    pickle.dump(scaler,f)