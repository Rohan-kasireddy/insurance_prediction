# 1 load scaler.pkl and model.pkl from artifacts folder
# 2 create a function to predict
import pickle
import numpy as np
class Insurance_prediction:
    def __init__(self):
        with open('artifacts/scaler.pkl','rb') as f:
            self.scaler=pickle.load(f)
        with open('artifacts/model.pkl','rb') as f:
            self.model=pickle.load(f)
    def prediction(self,Age,Annual_Income_LPA,Policy_Term_Years,Sum_Assured_Lakhs):
        input=np.array([[Age,Annual_Income_LPA,Policy_Term_Years,Sum_Assured_Lakhs]])
        input_scaled=self.scaler.transform(input)
        result=self.model.predict(input_scaled)
        return result[0]