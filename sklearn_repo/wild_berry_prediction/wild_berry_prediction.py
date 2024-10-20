#Implement algorithm for this task : https://www.kaggle.com/competitions/playground-series-s3e14/data

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn import ensemble
import matplotlib.pyplot as plt
from ml_scores import MLScores
import seaborn as sns
from xgboost import XGBRegressor

class DataCleanup:
    def read_clean_up(self,name):
        df = pd.read_csv(name)
        return df

    def identify_inputs(self,df):
        input_vars = []
        for column in df.columns:
            if column != 'yield' and column !='id':
                input_vars.append(column)
        return input_vars



class Prediction:
    def run_mainflow(self,ml_algo):
        datacleanup = DataCleanup()
        data_frame = datacleanup.read_clean_up('train.csv')
        input_variables = datacleanup.identify_inputs(data_frame)
        inputs = data_frame[input_variables]
        output = data_frame.filter(['yield'])
        print(inputs)
        print(output)

        X_train, X_test, Y_train, Y_test = train_test_split(inputs,output,test_size=0.3, random_state =0)
        pipeline = Pipeline(steps=[('scaler',StandardScaler()),('clf',ml_algo(random_state=0))])
        model_obj = pipeline.fit(X_train,Y_train['yield'])
        #SCORE VALUES
        train_score = pipeline.score(X_train, Y_train)
        model_score = pipeline.score(X_test, Y_test)
        print(f"Training model score : {train_score}")
        print(f"Final Score is: {model_score}")
        #predict
        Y_predict = model_obj.predict(X_test)


        scores = MLScores()
        scores.MAE_score(model_obj,pipeline, X_train, Y_train, X_test,Y_test)
        scores.max_error_score(model_obj,pipeline, X_train, Y_train, X_test,Y_test)
        scores.MSE_score(model_obj,pipeline, X_train, Y_train, X_test,Y_test)
        scores.mean_square_log_error(model_obj,pipeline, X_train, Y_train, X_test,Y_test)
        scores.Rsquared_score(model_obj,pipeline, X_train, Y_train, X_test,Y_test)
        scores.explained_variance_score(model_obj,pipeline, X_train, Y_train, X_test,Y_test)
        scores.mean_absolute_percentage_error(model_obj,pipeline, X_train, Y_train, X_test,Y_test)
        scores.mean_poisson_deviance_score(model_obj,pipeline, X_train, Y_train, X_test,Y_test)
        scores.mean_gamma_deviance_score(model_obj,pipeline, X_train, Y_train, X_test,Y_test)
        scores.mean_tweedie_deviance_score(model_obj,pipeline, X_train, Y_train, X_test,Y_test)
        scores.d2_tweedie_score_val(model_obj,pipeline, X_train, Y_train, X_test,Y_test)
        scores.mean_pinball_loss_val(model_obj,pipeline, X_train, Y_train, X_test,Y_test)
        scores.d2_pinball_score_val(model_obj,pipeline, X_train, Y_train, X_test,Y_test)
        scores.d2_absolute_error_val(model_obj,pipeline, X_train, Y_train, X_test,Y_test)


        #plot from sns
        fig,ax1=plt.subplots()
        sns.histplot(Y_predict,  color='blue', ax=ax1, label='Predicted value')
        sns.histplot(Y_test,color='red', ax=ax1, label='Actual value')
        plt.title('Actual vs Predicted values for flight price')
        plt.legend(['predicted','actual'])
        plt.show()

if __name__ == "__main__":
    prediction=Prediction()
    prediction.run_mainflow(XGBRegressor)
    prediction.run_mainflow(ensemble.RandomForestRegressor)
