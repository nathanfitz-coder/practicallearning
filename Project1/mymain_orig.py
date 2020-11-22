import pandas as pd
import xgboost as xgb
import numpy as np
import time
from scipy.stats import skew
from scipy.special import boxcox1p
from sklearn.linear_model import ElasticNet


def test_train(j):
    data = pd.read_csv('Ames_data.csv')
    testIDs = pd.read_table("project1_testIDs.dat",header=None,sep = ' ').values
    
    
    testidx = data.index.isin(testIDs[:,j])
    train = data[~testidx]
    test = data[testidx]
    
    train.to_csv("train.csv",index=False)
    test.drop(['Sale_Price'],axis=1).to_csv("test.csv",index=False)



def grid_elastic(train_x, train_y, test_x, test_y):
    alphas = []
    lambdas = []
    scores = []
    for i in range(1000):
        a = np.random.random()/1000
        l = np.random.random()
        alphas.append(a)
        lambdas.append(l)
        
        regr = ElasticNet(alpha=a, l1_ratio=l, random_state=42,normalize=True)
        #regr = make_pipeline(RobustScaler(), ElasticNet(alpha=a, l1_ratio=l, random_state=42))
        regr.fit(train_x, train_y)
        y_pred = regr.predict(test_x)
        scores.append(np.sqrt(np.mean(np.square(y_pred - test_y))))

    bestidx = np.argmin(np.array(scores))
    best_a = alphas[bestidx]
    best_l = lambdas[bestidx]
    best_score = scores[bestidx]
    return best_a, best_l, best_score



def all_test_splits(model='elastic'): 
    split_score = []
    
    if model == 'elastic':
        for split in range(0,10):
            test_train(split)
            a, l, returnf, best_score = main_elastic()
            split_score.append(best_score)
    else:  
        for split in range(0,10):
            test_train(split)
            returnf, best_score = main_xgb()
            split_score.append(best_score)
    return split_score

    
def main_elastic(a = 0.00004, l = 0.91, write_pred=False):
    train = pd.read_csv('train.csv',index_col='PID')
    test = pd.read_csv('test.csv',index_col='PID')
    
    alldata = pd.concat([train, test])
    alldata.Garage_Yr_Blt.fillna(alldata.Year_Built, inplace=True)
    
    #MSSubClass=The building class
    alldata['MS_SubClass'] = alldata['MS_SubClass'].apply(str)
    
    #Changing OverallCond into a categorical variable
    alldata['Overall_Cond'] = alldata['Overall_Cond'].astype(str)
    
    numeric_feats = alldata.dtypes[alldata.dtypes != "object"].index

    
    
    
    # Check the skew of all numerical features
    skewed_feats = alldata[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    #print("\nSkew in numerical features: \n")
    skewness = pd.DataFrame({'Skew' :skewed_feats})


    skewness = skewness[abs(skewness) > 0.75]
    #print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

    skewed_features = skewness.index
    lam = 0.15
    for feat in skewed_features:
        alldata[feat] = boxcox1p(alldata[feat], lam)
    
    #Adding total square footage
    alldata['Total_SF'] = alldata['Total_Bsmt_SF'] + alldata['First_Flr_SF'] + alldata['Second_Flr_SF']
    
    drop_vars = ['Street', 'Utilities',  'Condition_2', 'Roof_Matl', 'Heating', 'Pool_QC', 'Misc_Feature', 'Low_Qual_Fin_SF', 'Pool_Area', 'Longitude','Latitude']
    
    alldata = alldata.drop(columns=drop_vars)
    
    quant_vars = ["Lot_Frontage","Total_SF", "Lot_Area", "Mas_Vnr_Area", "BsmtFin_SF_2", "Bsmt_Unf_SF", "Total_Bsmt_SF", "Second_Flr_SF", 'First_Flr_SF', "Gr_Liv_Area", "Garage_Area", "Wood_Deck_SF", "Open_Porch_SF", "Enclosed_Porch", "Three_season_porch", "Screen_Porch", "Misc_Val"]
    
    for var in quant_vars:
        q95 = np.quantile(alldata[var],0.95)
        alldata[var][alldata[var]>q95] = q95
    
    
    alldata = pd.get_dummies(alldata,drop_first=True)
    
    train_x = alldata[alldata.index.isin(train.index)]
    test_x = alldata[alldata.index.isin(test.index)]
    
    # train_y = train_x.Sale_Price
    # test_y = test_x.Sale_Price
    train_y = train.Sale_Price
    test_y = test.Sale_Price
    
    
    
    train_x = train_x.drop(['Sale_Price'],axis=1)
    test_x = test_x.drop(['Sale_Price'],axis=1)
    
    train_y = np.log(train_y)
    test_y = np.log(test_y)
    
    # best_a, best_l, best_score = grid_elastic(train_x, train_y, test_x, test_y)
    
    # a = best_a
    # l = best_l
    
    regr = ElasticNet(alpha=a, l1_ratio=l, random_state=42,normalize=True)
    regr.fit(train_x, train_y)
    y_pred = regr.predict(test_x)
    returnf = pd.DataFrame(data=np.matrix.transpose(np.array([test_x.index.values,np.exp(y_pred)])),  columns=["PID", "Sale_Price"])
    if write_pred:
        #np.savetxt(fname='mysubmission1.txt',X=y_pred)
        returnf.astype({'PID': 'int32'}).to_csv('mysubmission1.txt',index=False)

    #return returnf

    return a, l, returnf, np.sqrt(np.mean(np.square(y_pred - test_y)))






def main_xgb(write_pred=False):
    train = pd.read_csv('train.csv',index_col='PID')
    test = pd.read_csv('test.csv',index_col='PID')
    
    alldata = pd.concat([train, test])
    alldata.Garage_Yr_Blt.fillna(alldata.Year_Built, inplace=True)
    
    #MSSubClass=The building class
    #alldata['MS_SubClass'] = alldata['MS_SubClass'].apply(str)
    
    #Changing OverallCond into a categorical variable
    #alldata['Overall_Cond'] = alldata['Overall_Cond'].astype(str)
    
    numeric_feats = alldata.dtypes[alldata.dtypes != "object"].index


    # Check the skew of all numerical features
    skewed_feats = alldata[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    #print("\nSkew in numerical features: \n")
    skewness = pd.DataFrame({'Skew' :skewed_feats})


    skewness = skewness[abs(skewness) > 0.75]
    #print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

    skewed_features = skewness.index
    lam = 0.15
    for feat in skewed_features:
        alldata[feat] = boxcox1p(alldata[feat], lam)
    
    #Adding total square footage
    alldata['Total_SF'] = alldata['Total_Bsmt_SF'] + alldata['First_Flr_SF'] + alldata['Second_Flr_SF']
    
    # drop_vars = ['Street', 'Utilities',  'Condition_2', 'Roof_Matl', 'Heating', 'Pool_QC', 'Misc_Feature', 'Low_Qual_Fin_SF', 'Pool_Area', 'Longitude','Latitude']
    
    # alldata = alldata.drop(columns=drop_vars)
    
    object_cols = list(alldata.dtypes[alldata.dtypes == 'object'].index)
    for col in object_cols:
        codes, uniques = pd.factorize(alldata[col])
        alldata[col]=codes
        
        
    quant_vars = ["Lot_Frontage","Total_SF", "Lot_Area", "Mas_Vnr_Area", "BsmtFin_SF_2", "Bsmt_Unf_SF", "Total_Bsmt_SF", "Second_Flr_SF", 'First_Flr_SF', "Gr_Liv_Area", "Garage_Area", "Wood_Deck_SF", "Open_Porch_SF", "Enclosed_Porch", "Three_season_porch", "Screen_Porch", "Misc_Val"]
    
    for var in quant_vars:
        q95 = np.quantile(alldata[var],0.95)
        alldata[var][alldata[var]>q95] = q95
    
    

    #alldata = pd.get_dummies(alldata,drop_first=True)

    train_x = alldata[alldata.index.isin(train.index)]
    test_x = alldata[alldata.index.isin(test.index)]
    
    # train_y = train_x.Sale_Price
    # test_y = test_x.Sale_Price
    train_y = train.Sale_Price
    test_y = test.Sale_Price
    
    
    
    train_x = train_x.drop(['Sale_Price'],axis=1)
    test_x = test_x.drop(['Sale_Price'],axis=1)
    
    train_y = np.log(train_y)
    test_y = np.log(test_y)
    

    
    
    model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                      learning_rate=0.05, max_depth=6, 
                      min_child_weight=1.7817, n_estimators=6000,
                      reg_alpha=0.4640, reg_lambda=0.8571,
                      subsample=0.5213, silent=1,
                      random_state =42, nthread = -1)
    
    
    
    model_xgb.fit(train_x, train_y, verbose=False)
    

    y_pred = model_xgb.predict(test_x)
    score=np.sqrt(np.mean(np.square(y_pred - test_y)))
    returnf = pd.DataFrame(data=np.matrix.transpose(np.array([test_x.index.values.astype(int),np.exp(y_pred)])),  columns=["PID", "Sale_Price"])
    if write_pred:
        #np.savetxt(fname='mysubmission2.txt',X=y_pred)
        returnf.astype({'PID': 'int32'}).to_csv('mysubmission2.txt',index=False)
    #split_score.append(np.sqrt(np.mean(np.square(y_pred - test_y))))
    
    
    return returnf, score



test_train(2)


tic = time.time()


returnf, score = main_xgb(write_pred=True)
a, l, returnf, score = main_elastic(write_pred=True)


toc = time.time()
difference = int(toc - tic)



# all_test_splits(model='xgbm')




