from sklearn import metrics
import sklearn
import pandas as pd

class MLScores:
    def __init__(self):
        pass
    def explained_variance_score(self,model_obj,pipeline, X_train, Y_train, X_test,Y_test):
        Y_predict = model_obj.predict(X_test)
        df_output = pd.DataFrame()
        df_output['Y_test'] = Y_test
        df_output['Y_predict'] = Y_predict
        #print(df_output)
        score = metrics.explained_variance_score(Y_test, Y_predict)
        print("Explained variance score", score)

    def max_error_score(self,model_obj,pipeline, X_train, Y_train, X_test,Y_test):
        Y_predict = model_obj.predict(X_test)
        df_output = pd.DataFrame()
        df_output['Y_test'] = Y_test
        df_output['Y_predict'] = Y_predict
        #print(df_output)
        score = metrics.max_error(Y_test, Y_predict)
        print("Max Error score", score)

    def MAE_score(self,model_obj,pipeline, X_train, Y_train, X_test,Y_test):
        Y_predict = model_obj.predict(X_test)
        df_output=pd.DataFrame()
        df_output['Y_test']=Y_test
        df_output['Y_predict']=Y_predict
        #print(df_output)
        norm_Y_test = Y_test-Y_test.min()/(Y_test.max()-Y_test.min())
        norm_Y_predict= Y_predict-Y_predict.min()/(Y_predict.max()-Y_predict.min())
        mae_score = metrics.mean_absolute_error(Y_test,Y_predict)
        print("MAE score", mae_score)

    def MSE_score(self,model_obj,pipeline, X_train, Y_train, X_test,Y_test):
        Y_predict = model_obj.predict(X_test)
        df_output = pd.DataFrame()
        df_output['Y_test'] = Y_test
        df_output['Y_predict'] = Y_predict
        #print(df_output)
        mse_score = metrics.mean_squared_error(Y_test, Y_predict)
        print("MSE score", mse_score)

    def mean_square_log_error(self,model_obj,pipeline, X_train, Y_train, X_test,Y_test):
        Y_predict = model_obj.predict(X_test)
        df_output = pd.DataFrame()
        df_output['Y_test'] = Y_test
        df_output['Y_predict'] = Y_predict
        #print(df_output)
        mse_score = metrics.mean_squared_log_error(Y_test, Y_predict)
        print("Mean square log error", mse_score)

    def mean_absolute_percentage_error(self,model_obj,pipeline, X_train, Y_train, X_test,Y_test):
        Y_predict = model_obj.predict(X_test)
        df_output = pd.DataFrame()
        df_output['Y_test'] = Y_test
        df_output['Y_predict'] = Y_predict
        #print(df_output)
        score = metrics.mean_absolute_percentage_error(Y_test, Y_predict)
        print("Mean absolute percentage error score", score)

    def Rsquared_score(self,model_obj,pipeline, X_train, Y_train, X_test,Y_test):
        Y_predict = model_obj.predict(X_test)
        df_output=pd.DataFrame()
        df_output['Y_test']=Y_test
        df_output['Y_predict']=Y_predict
        #print(df_output)
        r2_score = round(metrics.r2_score(Y_test,Y_predict),4)
        print("R2 score from metric", r2_score)


    def mean_poisson_deviance_score(self,model_obj,pipeline, X_train, Y_train, X_test,Y_test):
        Y_predict = model_obj.predict(X_test)
        df_output=pd.DataFrame()
        df_output['Y_test']=Y_test
        df_output['Y_predict']=Y_predict
        #print(df_output)
        score = round(metrics.mean_poisson_deviance(Y_test,Y_predict),4)
        print("Poisson deviance", score)

    def mean_gamma_deviance_score(self,model_obj,pipeline, X_train, Y_train, X_test,Y_test):
        Y_predict = model_obj.predict(X_test)
        df_output=pd.DataFrame()
        df_output['Y_test']=Y_test
        df_output['Y_predict']=Y_predict
        #print(df_output)
        score = round(metrics.mean_gamma_deviance(Y_test,Y_predict),4)
        print("Gamma deviance", score)

    def mean_tweedie_deviance_score(self,model_obj,pipeline, X_train, Y_train, X_test,Y_test):
        Y_predict = model_obj.predict(X_test)
        df_output=pd.DataFrame()
        df_output['Y_test']=Y_test
        df_output['Y_predict']=Y_predict
        #print(df_output)
        score = round(metrics.mean_tweedie_deviance(Y_test,Y_predict),4)
        print("Tweedie deviance", score)

    def d2_tweedie_score_val(self,model_obj,pipeline, X_train, Y_train, X_test,Y_test):
        Y_predict = model_obj.predict(X_test)
        df_output=pd.DataFrame()
        df_output['Y_test']=Y_test
        df_output['Y_predict']=Y_predict
        #print(df_output)
        score = round(metrics.d2_tweedie_score(Y_test,Y_predict),4)
        print("d2 Tweedie score", score)

    def mean_pinball_loss_val(self,model_obj,pipeline, X_train, Y_train, X_test,Y_test):
        Y_predict = model_obj.predict(X_test)
        df_output=pd.DataFrame()
        df_output['Y_test']=Y_test
        df_output['Y_predict']=Y_predict
        #print(df_output)
        score = round(metrics.mean_pinball_loss(Y_test,Y_predict),4)
        print("Mean pinball loss", score)

    def d2_pinball_score_val(self,model_obj,pipeline, X_train, Y_train, X_test,Y_test):
        Y_predict = model_obj.predict(X_test)
        df_output=pd.DataFrame()
        df_output['Y_test']=Y_test
        df_output['Y_predict']=Y_predict
        #print(df_output)
        score = round(metrics.d2_pinball_score(Y_test,Y_predict),4)
        print("D2 pinball score", score)

    def d2_absolute_error_val(self,model_obj,pipeline, X_train, Y_train, X_test,Y_test):
        Y_predict = model_obj.predict(X_test)
        df_output=pd.DataFrame()
        df_output['Y_test']=Y_test
        df_output['Y_predict']=Y_predict
        #print(df_output)
        score = round(metrics.d2_absolute_error_score(Y_test,Y_predict),4)
        print("D2 absolute error score", score)

if __name__ == '__main__':
    ml_scores = MLScores()
    print("MSE score", ml_scores.MSE_score())


