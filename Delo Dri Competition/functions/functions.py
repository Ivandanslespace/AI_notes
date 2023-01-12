import config
import numpy as np
import pandas as pd

##########################################################################################################################
############################################################ Load Data ###################################################
##########################################################################################################################

def load (data_path = config.data_path, label = config.label, drop_first_row=True):
    data = pd.read_excel(data_path, index_col="TRIMESTRE")
    if "dtf_per_trt" in data.columns:
        data.drop(columns=["dtf_per_trt"], inplace=True)
    data.fillna(0, inplace=True)
    if drop_first_row:
        data = data.drop(["2008Q4"])
    data[label] = data[label]*100

    CHR2 = data[data['CHRONIQUE']=='CHR2']
    CHR8 = data[data['CHRONIQUE']=='CHR8']
    CHR_total = data[data['CHRONIQUE']=='Totale']

    return CHR2, CHR8, CHR_total

def load_csv (data_path = config.data_path_csv, label = config.label, drop_first_row=True):
    data = pd.read_csv(data_path, index_col="TRIMESTRE")
    if "dtf_per_trt" in data.columns:
        data.drop(columns=["dtf_per_trt"], inplace=True)
    data.fillna(0, inplace=True)
    if drop_first_row:
        data = data.drop(["2008Q4"])
    data[label] = data[label]*100

    CHR2 = data[data['CHRONIQUE']=='CHR2']
    CHR8 = data[data['CHRONIQUE']=='CHR8']
    CHR_total = data[data['CHRONIQUE']=='Totale']

    return CHR2, CHR8, CHR_total

def load_csv_3_groups(data_type="normal"):
    if data_type=="normal":
        CHR2 = pd.read_csv("Data/CHR2.csv", index_col=0)
        CHR2[config.label] = CHR2[config.label]*100
        CHR8 = pd.read_csv("Data/CHR8.csv", index_col=0)
        CHR8[config.label] = CHR8[config.label]*100
        CHRt = pd.read_csv("Data/CHRt.csv", index_col=0)
        CHRt[config.label] = CHRt[config.label]*100
    if data_type=="Poly":
        CHR2 = pd.read_csv("Data/CHR2_Poly.csv", index_col=0)
        CHR2[config.label] = CHR2[config.label]*100
        CHR8 = pd.read_csv("Data/CHR8_Poly.csv", index_col=0)
        CHR8[config.label] = CHR8[config.label]*100
        CHRt = pd.read_csv("Data/CHRt_Poly.csv", index_col=0)
        CHRt[config.label] = CHRt[config.label]*100  
    if data_type=="DeTemp":
        CHR2 = pd.read_csv("Data/CHR2_DeTemp.csv", index_col=0)
        CHR2[config.label] = CHR2[config.label]*100
        CHR8 = pd.read_csv("Data/CHR8_DeTemp.csv", index_col=0)
        CHR8[config.label] = CHR8[config.label]*100
        CHRt = pd.read_csv("Data/CHRt_DeTemp.csv", index_col=0)
        CHRt[config.label] = CHRt[config.label]*100  
    if data_type=="Poly+DeTemp":
        CHR2 = pd.read_csv("Data/CHR2_Poly+DeTemp.csv", index_col=0)
        CHR2[config.label] = CHR2[config.label]*100
        CHR8 = pd.read_csv("Data/CHR8_Poly+DeTemp.csv", index_col=0)
        CHR8[config.label] = CHR8[config.label]*100
        CHRt = pd.read_csv("Data/CHRt_Poly+DeTemp.csv", index_col=0)
        CHRt[config.label] = CHRt[config.label]*100  

    return CHR2, CHR8, CHRt

from sklearn.model_selection import train_test_split
def train_test_data(data,
                    variable_chosen,
                    label=config.label,
                    test_set=config.test_set):
    # final_variables_list = data.drop(columns=["CHRONIQUE", "DR", "dtf_per_trt"]).columns.values.tolist()
    y = data[label]
    X = data[variable_chosen]

    # X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=config.train_size, random_state = config.TrainTestSplit_random_state)
    X_train = X.drop(test_set)
    y_train = y.drop(test_set)
    X_test = X.loc[test_set]
    y_test = y.loc[test_set]
    
    return X, y, X_train, X_test, y_train, y_test

##########################################################################################################################
############################################################ Features Generation #########################################
##########################################################################################################################

def Binary_PolynomialFeatures(colNames, degree, features):
    """
    Continuous variable two-variable polynomial derived functions
    
    :param colNames: Names of columns involved in cross derivation
    :param degree: Polynomial of highest order
    :param features: Raw data set
    
    :return：New features and new column names after cross-derivation
    """
    
    
    # Create empty list memory
    colNames_new_l = []
    features_new_l = []
    
    # Extraction of features requiring polynomial derivation
    features = features[colNames]
    
    # Combining polynomial features one by one
    for col_index, col_name in enumerate(colNames):
        for col_sub_index in range(col_index+1, len(colNames)):
            col_temp = [col_name] + [colNames[col_sub_index]]
            array_new_temp = PolynomialFeatures(degree=degree, include_bias=False).fit_transform(features[col_temp])
            features_new_l.append(pd.DataFrame(array_new_temp[:, 2:]))
    
            # Create the names of the derived polynomial features one by one
            for deg in range(2, degree+1):
                for i in range(deg+1):
                    col_name_temp = col_temp[0] + '**' + str(deg-i) + '*'+ col_temp[1] + '**' + str(i)
                    colNames_new_l.append(col_name_temp)
            
    
    # Splicing the new feature matrix
    features_new = pd.concat(features_new_l, axis=1)
    features_new.columns = colNames_new_l
    colNames_new = colNames_new_l
    
    return features_new, colNames_new

def Multi_PolynomialFeatures(colNames, degree, features):
    """
    Continuous variable multivariate polynomial derived functions
    
    :param colNames: Names of columns involved in cross derivation
    :param degree: Polynomial of highest order
    :param features: Raw data set
    
    :return：New features and new column names after cross-derivation
    """
    
    
    # Create empty list container
    colNames_new_l = []
    
    # Calculate the number of characteristics brought into the polynomial calculation
    n = len(colNames)
    
    # Extraction of features requiring polynomial derivation
    features = features[colNames]
    
    # Perform polynomial feature combination
    array_new_temp = PolynomialFeatures(degree=degree, include_bias=False).fit_transform(features)
    # Selection of derived features
    array_new_temp = array_new_temp[:, n:]
    
    
    # Create a list of column names
    deg = 2
    while deg <= degree:
        m = 1
        a1 = range(deg, -1, -1)
        a2 = []
        while m < n:
            a1 = list(product(a1, range(deg, -1, -1)))
            if m > 1:
                for i in a1:
                    i_temp = list(i[0])
                    i_temp.append(i[1])
                    a2.append(i_temp)
            m += 1
        a2 = np.array(a2)
        a3 = a2[a2.sum(1) == deg]
        
        for i in a3:
            colNames_new_l.append('&'.join(colNames) + '_' + ''.join([str(i) for i in i]))    
        
        deg += 1
    
    # Splicing the new feature matrix
    features_new = pd.DataFrame(array_new_temp, columns=colNames_new_l)
    colNames_new = colNames_new_l
    
    return features_new, colNames_new

def Polynomial_Features(colNames, 
                        degree, 
                        X_train, 
                        X_test, 
                        multi=False):   
    
    """
    Polynomial characteristic derived function
    
    :param colNames: the names of the columns involved in the cross derivation
    :param degree: the highest order of the polynomial
    :param X_train: training set features
    :param X_test: test set features
    :param multi: whether to perform multivariate polynomial group derivation
    
    :return: new features and new column names after polynomial derivation
    """
    if multi == False:
        features_train_new, colNames_train_new = Binary_PolynomialFeatures(colNames=colNames, degree=degree, features=X_train)
        features_test_new, colNames_test_new = Binary_PolynomialFeatures(colNames=colNames, degree=degree, features=X_test)
    else:
        features_train_new, colNames_train_new = Multi_PolynomialFeatures(colNames=colNames, degree=degree, features=X_train)
        features_test_new, colNames_test_new = Multi_PolynomialFeatures(colNames=colNames, degree=degree, features=X_test)
        
    assert colNames_train_new  == colNames_test_new
    return features_train_new, features_test_new, colNames_train_new, colNames_test_new

##########################################################################################################################
############################################################ Features Selection ##########################################
##########################################################################################################################
from sklearn.decomposition import PCA
def pca_transformation_3(data, new_dataframe, variable_index, variables_list, n_components=3):
    variable= data[variables_list]
    pca = PCA(n_components=n_components)
    new_dataframe[["v"+str(variable_index+1)+"_1", "v"+str(variable_index+1)+"_2", "v"+str(variable_index+1)+"_3"]] = pca.fit_transform(variable)

def pca_transformation_4(data, new_dataframe, variable_index, variables_list, n_components=4):
    variable= data[variables_list]
    pca = PCA(n_components=n_components)
    new_dataframe[["v"+str(variable_index+1)+"_1", "v"+str(variable_index+1)+"_2", 
                    "v"+str(variable_index+1)+"_3", "v"+str(variable_index+1)+"_4"]] = pca.fit_transform(variable)

def pca_pipeline(data, variables_list, PCA_n_components, groupby_columns, complement_data_columns):
    features = data[variables_list]
    mmscaler = MinMaxScaler()
    trans_data = mmscaler.fit_transform(features)
    scaler = StandardScaler()
    trans_data = scaler.fit_transform(trans_data)
    trans_data = pd.DataFrame(trans_data, columns=variables_list)

    pca_variables = pd.DataFrame()
    for index, variable_list in enumerate(groupby_columns):
        if PCA_n_components == 3:
            pca_transformation_3(trans_data, pca_variables, index, variable_list)
        if PCA_n_components == 4:
            pca_transformation_4(trans_data, pca_variables, index, variable_list)

    pca_variables.index = data.index
    pca_variables = pd.concat([pca_variables, data[complement_data_columns]], axis=1)
    pca_variables[config.label] = data[config.label]

    X, y, X_train, X_test, y_train, y_test = train_test_data(pca_variables, pca_variables.columns.drop("DR"))

    return X, y, X_train, X_test, y_train, y_test

def mRMR_data_name(mrmr_K, type):
    if type=='normal':
        data_name = "mRMR_"+str(mrmr_K)
    if type=='Poly':
        data_name = "Poly_mRMR_"+str(mrmr_K)
    if type=='DeTemp':
        data_name = "DeTemp_mRMR_"+str(mrmr_K)
    if type=='Poly+DeTemp':
        data_name = "Poly_DeTemp_mRMR_"+str(mrmr_K)
    return data_name

##########################################################################################################################
############################################################ Simple Model ################################################
##########################################################################################################################

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
def Linear_Regression(X, y, X_train, y_train, X_test, y_test, result,
                        data_name, Group_name,
                        print_score=config.print_score, ploton=config.ploton, standardizer="S",
                        features_mRMR=np.nan):
    model_name="LR"

    
    # Building training pipeline
    if standardizer == "S":
        pipe = make_pipeline(StandardScaler(),
                            LinearRegression())
    if standardizer == "M":
        pipe = make_pipeline(MinMaxScaler(),
                            LinearRegression())
    if standardizer == "S+M":
        pipe = make_pipeline(StandardScaler(),
                            MinMaxScaler(),
                            LinearRegression())

    model = pipe.fit(X_train, y_train) # Fit

    # CV score
    cv = cross_val_score(
        estimator=pipe,
        X=X_train,
        y=y_train,
        scoring=config.scoring,
        cv=len(X_train))

    # Chcek CV performance on train dataset
    train_RMSE = -cv.mean()
    if print_score:
        print("neg_root_mean_squared_error on train set is: ",train_RMSE)
    if ploton:
        best_estomator_plot_MSE(model, X, y, data_name, standardizer, Group_name, model_name)

    # Check model performance on test dataset
    test_RMSE = mean_squared_error(y_test, model.predict(X_test), squared=False)

    # Write down results
    # series = pd.Series({"Data": data_name, "Standardizer": standardizer, 
    #                     "Model":model_name,"Group":Group_name, 
    #                     "Train_RMSE":train_RMSE,  "Test_RMSE":test_RMSE, "num_features" : int(X.shape[1])})
        # result = result.append(series, ignore_index=True)
    result_model = {"Data": data_name, "Standardizer": standardizer, 
                    "Model":model_name,"Group":Group_name, 
                    "Train_RMSE":train_RMSE,  "Test_RMSE":test_RMSE,
                    "features_mRMR" : features_mRMR, "num_features" : int(X.shape[1])}
    series = pd.Series(result_model)
    result = result.append(series, ignore_index=True)

    return result

def Linear_Regression_Elas(X, y, X_train, y_train, X_test, y_test, result,
                            data_name, Group_name,
                            print_score=config.print_score, ploton=config.ploton, standardizer="S",
                            features_mRMR=np.nan):
    model_name="LR+Elas"
    
    if standardizer == "S":
        pipe = make_pipeline(StandardScaler(),
                            ElasticNet())
    if standardizer == "M":
        pipe = make_pipeline(MinMaxScaler(),
                            ElasticNet())
    if standardizer == "S+M":
        pipe = make_pipeline(StandardScaler(),
                            MinMaxScaler(),
                            ElasticNet())

    param_grid = [{'elasticnet__alpha':np.arange(0.0, 1.1, 0.1).tolist(), 
                'elasticnet__l1_ratio':np.arange(0.0, 1.1, 0.1).tolist()}]

    result = GridSearchCV_pipeline(X, y, X_train, X_test, y_train, y_test, 
                                                    pipe, param_grid, result,
                                                    data_name, standardizer, Group_name, model_name,
                                                    features_mRMR=features_mRMR)

    return result

def Polynomial_Regression_Elas(X, y, X_train, y_train, X_test, y_test, result,
                            data_name, Group_name,
                            print_score=config.print_score, ploton=config.ploton, standardizer="S", order=2,
                            features_mRMR=np.nan):
    model_name="Poly2+Elas"
    
    if standardizer == "S":
        pipe = make_pipeline(PolynomialFeatures(degree=order), 
                            StandardScaler(),
                            ElasticNet())
    if standardizer == "M":
        pipe = make_pipeline(PolynomialFeatures(degree=order), 
                            MinMaxScaler(),
                            ElasticNet())
    if standardizer == "S+M":
        pipe = make_pipeline(PolynomialFeatures(degree=order), 
                            StandardScaler(),
                            MinMaxScaler(),
                            ElasticNet())



    param_grid = [{'elasticnet__alpha':np.arange(0.0, 1.1, 0.1).tolist(), 
        'elasticnet__l1_ratio':np.arange(0.0, 1.1, 0.1).tolist()}]

    result = GridSearchCV_pipeline(X, y, X_train, X_test, y_train, y_test, 
                                                    pipe, param_grid, result,
                                                    data_name, standardizer, Group_name, model_name,
                                                    features_mRMR=features_mRMR)

    return result
    
##########################################################################################################################
############################################################ Ensemble Model ##############################################
##########################################################################################################################
from sklearn.ensemble import RandomForestRegressor as RFR
import xgboost as xgb
from sklearn.model_selection import KFold
from hyperopt import hp, fmin, tpe, Trials, partial
from hyperopt.early_stop import no_progress_loss

def Random_Forest_Regresssion(X, y, X_train, y_train, X_test, y_test, result,
                            data_name, Group_name, mrmr_K,
                            print_score=config.print_score, ploton=config.ploton, standardizer="S",
                            features_mRMR=np.nan):
    model_name = "RFR"
    def hyperopt_objective(params):
        
        #Define the evaluator
        #Parameters that need to be searched need to be indexed from the input dictionary
        #Parameters that don't need to be searched can be some value that is set
        #Adjust parameter types before parameters that require integers
        reg = RFR(n_estimators = int(params["n_estimators"])
                ,max_depth = int(params["max_depth"])
                ,max_features = int(params["max_features"])
                ,min_impurity_decrease = params["min_impurity_decrease"]
                ,random_state=1412
                ,verbose=False
                ,n_jobs=-1)
        
        #Cross-validation results, output negative root mean square error (-RMSE)
        cv = KFold(n_splits=len(X_train),shuffle=True,random_state=1412)
        validation_loss = cross_validate(reg,X_train,y_train
                                        ,scoring="neg_root_mean_squared_error"
                                        ,cv=cv
                                        ,verbose=False
                                        ,n_jobs=-1
                                        ,error_score='raise'
                                        )
        
        #The final output, since it can only take the minimum value, must find the absolute value for (-RMSE)
        # to solve for the combination of parameters corresponding to the minimum RMSE
        return np.mean(abs(validation_loss["test_score"]))

    param_grid_simple = {'n_estimators': hp.quniform("n_estimators",50,150,1)
                        , 'max_depth': hp.quniform("max_depth",2,10,1)
                        , "max_features": hp.quniform("max_features",1,mrmr_K,1)
                        , "min_impurity_decrease":hp.quniform("min_impurity_decrease",0,3,1)
                        }

    def param_hyperopt(max_evals=100):
        
        #Save iterative process
        trials = Trials()
        
        # Set early stop
        early_stop_fn = no_progress_loss(50)
        
        #DefinitionProxyModel
        #algo = partial(tpe.suggest, n_startup_jobs=20, n_EI_candidates=50)
        params_best = fmin(hyperopt_objective #目标函数
                        , space = param_grid_simple #parameter space
                        , algo = tpe.suggest # Which proxy model do you want?
                        #, algo = algo
                        , max_evals = max_evals # number of iterations allowed
                        , verbose=True
                        , trials = trials
                        , early_stop_fn = early_stop_fn
                        )
        
        #Print the optimal parameters, fmin will automatically print the best score
        print("\n","\n","best params: ", params_best,
            "\n")
        return params_best, trials

    def hyperopt_validation(params, X, y):    
        reg = RFR(n_estimators = int(params["n_estimators"])
                ,max_depth = int(params["max_depth"])
                ,max_features = int(params["max_features"])
                ,min_impurity_decrease = params["min_impurity_decrease"]
                ,random_state=1412
                ,verbose=False
                ,n_jobs=-1
                )
        cv = KFold(n_splits=len(X_test),shuffle=True,random_state=1412)
        validation_loss = cross_validate(reg,X,y
                                        ,scoring="neg_root_mean_squared_error"
                                        ,cv=cv
                                        ,verbose=False
                                        ,n_jobs=-1
                                        )
        return np.mean(abs(validation_loss["test_score"]))

    params_best, trials = param_hyperopt(150) #1% of space size

    train_RMSE = hyperopt_validation(params_best, X_train, y_train)
    test_RMSE = hyperopt_validation(params_best, X_test, y_test)

    result_model = {"Data": data_name, "Standardizer": standardizer, 
                    "Model":model_name,"Group":Group_name, 
                    "Train_RMSE":train_RMSE,  "Test_RMSE":test_RMSE,
                    "num_features" : int(X.shape[1])}
    params = {"params":params_best}
    result_model.update(params)
    features_mRMR = {"features_mRMR" : features_mRMR}
    result_model.update(features_mRMR)

    series = pd.Series(result_model)
    result = result.append(series, ignore_index=True)
    
    return result

def XGBoost_Regression(X, y, X_train, y_train, X_test, y_test, result,
                            data_name, Group_name,
                            print_score=config.print_score, ploton=config.ploton, standardizer="S",
                            features_mRMR=np.nan):
    model_name = "XGBoost"

    data_xgb = xgb.DMatrix(X_train,y_train)
    data_xgb_test = xgb.DMatrix(X_test,y_test)

    def hyperopt_objective(params):
        paramsforxgb = {"eta":params["eta"]
                        ,"booster":params["booster"]
                        ,"colsample_bytree":params["colsample_bytree"]
                        ,"colsample_bynode":params["colsample_bynode"]
                        ,"gamma":params["gamma"]
                        ,"lambda":params["lambda"]
                        ,"min_child_weight":params["min_child_weight"]
                        ,"max_depth":int(params["max_depth"])
                        ,"subsample":params["subsample"]
                        ,"objective":params["objective"]
                        ,"rate_drop":params["rate_drop"]
                        ,"nthread":14
                        ,"verbosity":0
                        ,"seed":1412}
        result = xgb.cv(paramsforxgb,data_xgb, seed=1412, metrics=("rmse")
                        ,num_boost_round=int(params["num_boost_round"]))
        return result.iloc[-1,2]

    param_grid_simple = {'num_boost_round': hp.quniform("num_boost_round",30,150,10)
                        ,"eta": hp.quniform("eta",1.05,3.05,0.05)
                        ,"booster":hp.choice("booster",["gbtree","dart"])
                        ,"colsample_bytree":hp.quniform("colsample_bytree",0.3,1,0.05)
                        ,"colsample_bynode":hp.quniform("colsample_bynode",0.1,1,0.05)
                        ,"gamma":hp.quniform("gamma",1e6,1e7,1e6)
                        ,"lambda":hp.quniform("lambda",2,5,0.2)
                        ,"min_child_weight":hp.quniform("min_child_weight",0,50,2)
                        ,"max_depth":hp.choice("max_depth",[*range(2,10,1)])
                        ,"subsample":hp.quniform("subsample",0.1,1,0.05)
                        ,"objective":hp.choice("objective",["reg:squarederror","reg:squaredlogerror"])
                        ,"rate_drop":hp.quniform("rate_drop",0.0,1,0.05)
                        }

    def param_hyperopt(max_evals=100):
        
        #Save iterative process
        trials = Trials()
        
        # Set early stop
        early_stop_fn = no_progress_loss(50)
        
        #DefinitionProxyModel
        params_best = fmin(hyperopt_objective
                        , space = param_grid_simple
                        , algo = tpe.suggest
                        , max_evals = max_evals
                        , verbose=True
                        , trials = trials
                        , early_stop_fn = early_stop_fn
                        )
        
        print("\n","\n","best params: ", params_best,
                "\n")
        return params_best, trials

    def hyperopt_validation(params, data):
        paramsforxgb = {"eta":params["eta"]
                        ,"booster":"dart"
                        ,"colsample_bytree":params["colsample_bytree"]
                        ,"colsample_bynode":params["colsample_bynode"]
                        ,"gamma":params["gamma"]
                        ,"lambda":params["lambda"]
                        ,"min_child_weight":params["min_child_weight"]
                        ,"max_depth":int(params["max_depth"])
                        ,"subsample":params["subsample"]
                        ,"rate_drop":params["rate_drop"]
                        ,"nthred":14
                        ,"verbosity":0
                        ,"seed":1412}
        result = xgb.cv(paramsforxgb,data, seed=1412, metrics=("rmse")
                        ,num_boost_round=int(params["num_boost_round"]))
        return result.iloc[-1,2]

    params_best, trials = param_hyperopt(150) # Training
    train_RMSE = hyperopt_validation(params_best, data_xgb)
    test_RMSE = hyperopt_validation(params_best, data_xgb_test)

    # Update resulat table
    result_model = {"Data": data_name, "Standardizer": standardizer, 
                    "Model":model_name,"Group":Group_name, 
                    "Train_RMSE":train_RMSE,  "Test_RMSE":test_RMSE,
                    "num_features" : int(X.shape[1])}
    params = {"params":params_best}
    result_model.update(params)
    features_mRMR = {"features_mRMR" : features_mRMR}
    result_model.update(features_mRMR)

    series = pd.Series(result_model)
    result = result.append(series, ignore_index=True)
    
    return result

##########################################################################################################################
############################################################ Trainning Process ###########################################
##########################################################################################################################
from sklearn.model_selection import GridSearchCV
def GridSearchCV_pipeline(X, y, X_train, X_test, y_train, y_test, 
                    pipe, param_grid, result_df,
                    data_name, standardizer_name, Group_name, model_name, features_mRMR=np.nan,
                    scoring=config.scoring):

                        model = GridSearchCV(
                            estimator=pipe,
                            param_grid=param_grid,
                            scoring= scoring,
                            cv=len(X_train))

                        model.fit(X_train, y_train)

                        train_RMSE, test_RMSE = performance_check_save(model, X, y, X_test, y_test,
                                                                                        data_name, standardizer_name, Group_name, model_name)

                        # Write down results
                        result_model = {"Data": data_name, "Standardizer": standardizer_name, 
                                        "Model":model_name,"Group":Group_name, 
                                        "Train_RMSE":train_RMSE,  "Test_RMSE":test_RMSE,
                                        "params" : model.best_params_, "features_mRMR" : features_mRMR, 
                                        "num_features" : int(X.shape[1])}
                        # params = {"params" : model.best_params_}
                        # result_model.update(params)

                        # features_mRMR = {"features_mRMR" : features_mRMR}
                        # result_model.update(features_mRMR)

                        # num_feautres = {"num_features" : int(X.shape[1])}
                        # result_model.update(num_feautres)

                        series = pd.Series(result_model)
                        result_df = result_df.append(series, ignore_index=True)
                        
                        return result_df

from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
import plotly_express as px 
import plotly.graph_objects as go
def best_estomator_plot_MSE(model, X_toPredict, y_True, data_name, standardizer_name, 
                            group_name, model_name, 
                            print_score=False, ploton=config.ploton):
    y_predict = model.predict(X_toPredict)

    if ploton:
        # plt.figure(figsize=(len(X_toPredict),3))

        # plt.plot(y_predict,"x-",label="Prediction")
        # plt.plot(y_True,"+-",label="True Value")
        # plt.legend()
        # plt.title(data_name+"_"+standardizer_name+"_"+group_name+"_"+model_name)

        # plt.show()
        trace1 = go.Scatter(
            x=y_True.index,
            y=y_predict,
            mode="lines",
            name="Prediction",
            opacity=0.6
        )

        trace2 = go.Scatter(
            x=y_True.index,
            y=y_True,
            mode="lines",
            name="True Value",
            opacity=0.6
        )

        fig = go.Figure(data=[trace1, trace2])
        fig.update_layout(width=900, height=300, title_text=data_name+"_"+standardizer_name+"_"+group_name+"_"+model_name)
        fig.show()

    if print_score:
        print("Mean Squared Error is ", mean_squared_error(y_True, y_predict, squared=False))

def best_model_result(cv_model):
    # showing best model
    # print("------------------------------------------------------------------------")
    print(-cv_model.best_score_)
    # print("------------------------------------------------------------------------")
    print(cv_model.best_params_)
    # print("------------------------------------------------------------------------")
    print(cv_model.best_estimator_)

def performance_check_save(model, 
                           X, y, X_test, y_test,
                           data_name, standardizer_name, Group_name, model_name, 
                           ploton=config.ploton, print_score=config.print_score):
    # Chcek CV performance on train dataset
    train_RMSE = -model.best_score_

    if print_score:
        print("------------------------------------------------------------------------")
        print("neg_root_mean_squared_error on CV is: ",train_RMSE)
        best_model_result(model)
    if ploton:
        best_estomator_plot_MSE(model.best_estimator_, X, y, data_name, standardizer_name, Group_name, model_name)

    # Check model performance on test dataset
    test_RMSE = mean_squared_error(y_test, model.predict(X_test), squared=False)

    return train_RMSE, test_RMSE