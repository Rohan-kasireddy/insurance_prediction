# 1 load Raw data
# 2 identifying x and y(i/o)
# 3 split data into train and test
import pandas as pd
from sklearn.model_selection import train_test_split
def load_and_split_data():
    df=pd.read_csv('../data/raw/insurance_data.csv')
    x=df[['Age','Annual_Income_LPA','Policy_Term_Years','Sum_Assured_Lakhs']]
    y=df['Annual_Premium_Thousands']
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
    return x_train,x_test,y_train,y_test
