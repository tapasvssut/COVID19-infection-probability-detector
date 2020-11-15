import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression

if __name__ == "__main__":
    df = pd.read_csv('corona_data.csv')
    
    X = df[['Fever','Body_Pain','Age','Runny_Nose','Difficulty_Breath']]
    y = df['Infection_Probability']
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
    clf = LogisticRegression()
    clf.fit(X_train,y_train)
    file = open('ML_model.pkl','wb')
    pickle.dump(clf,file)
    file.close()
    