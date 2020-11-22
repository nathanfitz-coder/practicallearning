import pandas as pd
import xgboost as xgb
import numpy as np
import time
from scipy.stats import skew
from scipy.special import boxcox1p
from sklearn.linear_model import ElasticNet
from sklearn import preprocessing

def test_train(j):
    data = pd.read_csv('Ames_data.csv')
    testIDs = pd.read_table("project1_testIDs.dat",header=None,sep = ' ').values
    
    
    testidx = data.index.isin(testIDs[:,j])
    train = data[~testidx]
    test = data[testidx]
    
    train.to_csv("train.csv",index=False)
    test.drop(['Sale_Price'],axis=1).to_csv("test.csv",index=False)
    #test.to_csv("test.csv",index=False)


class LabelEncoderExt(object):
    def __init__(self):
        """
        It differs from LabelEncoder by handling new classes and providing a value for it [Unknown]
        Unknown will be added in fit and transform will take care of new item. It gives unknown class id
        """
        self.label_encoder = preprocessing.LabelEncoder()
        # self.classes_ = self.label_encoder.classes_

    def fit(self, data_list):
        """
        This will fit the encoder for all the unique values and introduce unknown value
        :param data_list: A list of string
        :return: self
        """
        self.label_encoder = self.label_encoder.fit(list(data_list) + ['Unknown'])
        self.classes_ = self.label_encoder.classes_

        return self

    def transform(self, data_list):
        """
        This will transform the data_list to id list where the new values get assigned to Unknown class
        :param data_list:
        :return:
        """
        new_data_list = list(data_list)
        for unique_item in np.unique(data_list):
            if unique_item not in self.label_encoder.classes_:
                new_data_list = ['Unknown' if x==unique_item else x for x in new_data_list]

        return self.label_encoder.transform(new_data_list)
    
    
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


def elastic_prepro(data,trainframe=None,params={}, settype='train'):
    returnparams={}
    returndata = data.copy()
    quant_vars = ["Lot_Frontage","Total_SF", "Lot_Area", "Mas_Vnr_Area", "BsmtFin_SF_2", "Bsmt_Unf_SF", "Total_Bsmt_SF", "Second_Flr_SF", 'First_Flr_SF', "Gr_Liv_Area", "Garage_Area", "Wood_Deck_SF", "Open_Porch_SF", "Enclosed_Porch", "Three_season_porch", "Screen_Porch", "Misc_Val"]
    drop_vars = ['Street', 'Utilities',  'Condition_2', 'Roof_Matl', 'Heating', 'Pool_QC', 'Misc_Feature', 'Low_Qual_Fin_SF', 'Pool_Area', 'Longitude','Latitude']
    
    returndata = returndata.drop(columns=drop_vars)
    
        #MSSubClass=The building class
    returndata['MS_SubClass'] = returndata['MS_SubClass'].apply(str)
    
    #Changing OverallCond into a categorical variable
    returndata['Overall_Cond'] = returndata['Overall_Cond'].astype(str)
    
    
    
    returndata.Garage_Yr_Blt.fillna(data.Year_Built, inplace=True)
    
    #Adding total square footage
    returndata['Total_SF'] = returndata['Total_Bsmt_SF'] + returndata['First_Flr_SF'] + returndata['Second_Flr_SF']
    lam = 0.15
    if settype=='train':
        quantiles={}
        for var in quant_vars:
            q95 = np.quantile(returndata[var],0.95)
            returndata[var][returndata[var]>q95] = q95
            quantiles[var]=q95
        
        returnparams['quants']=quantiles
        
        #numeric_feats = returndata.dtypes[returndata.dtypes != "object"].index
        numeric_feats = returndata.dtypes[returndata.dtypes != "object"].index.tolist()
        # Check the skew of all numerical features
        skewed_feats = returndata[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
        #print("\nSkew in numerical features: \n")
        skewness = pd.DataFrame({'Skew' :skewed_feats})
        skewness = skewness[abs(skewness) > 0.75]
        #print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))
        skewed_features = skewness.index
        returnparams['skewed_features']=skewed_features
        
        for feat in skewed_features:
            returndata[feat] = boxcox1p(returndata[feat], lam)
    
        #get dummies for categorical features
        returndata = pd.get_dummies(returndata)

    if settype=='test':
        for var in quant_vars:
            q95 = params['quants'][var]
            returndata[var][returndata[var]>q95] = q95
            
        skewed_features=params['skewed_features']
        
        for feat in skewed_features:
            returndata[feat] = boxcox1p(returndata[feat], lam)
        
        #get dummies for categorical features and make it consistent with train dataframe
        returndata = pd.get_dummies(returndata)
        returndata = returndata.reindex(columns = trainframe.columns, fill_value=0)
        
        
        return returndata
        
    return returnparams, returndata
    
def main_elastic(a = 0.00004, l = 0.91, write_pred=False):
    train = pd.read_csv('train.csv',index_col='PID')
    test = pd.read_csv('test.csv',index_col='PID')
    
    train_y = train.Sale_Price
    #test_y = test.Sale_Price
    
    
    
    train = train.drop(['Sale_Price'],axis=1)
    #test = test.drop(['Sale_Price'],axis=1)
    
    train_y = np.log(train_y)
    #test_y = np.log(test_y)
    
    
    

    
    preparams, train_x = elastic_prepro(train)
    test_x = elastic_prepro(test,trainframe=train_x,params=preparams,settype='test')
    
    object_cols = list(train_x.dtypes[train_x.dtypes == 'object'].index)
    for col in object_cols:
        ohe = preprocessing.OneHotEncoder(handle_unknown='ignore')
        ohe.fit(train_x[col])
        train_x[col]=ohe.transform(train_x[col])
        test_x[col]=ohe.transform(test_x[col])
    
    
    

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

    return a, l, returnf #, np.sqrt(np.mean(np.square(y_pred - test_y)))



def xgb_prepro(data,params={}, settype='train'):
    returnparams={}
    returndata = data.copy()
    quant_vars = ["Lot_Frontage","Total_SF", "Lot_Area", "Mas_Vnr_Area", "BsmtFin_SF_2", "Bsmt_Unf_SF", "Total_Bsmt_SF", "Second_Flr_SF", 'First_Flr_SF', "Gr_Liv_Area", "Garage_Area", "Wood_Deck_SF", "Open_Porch_SF", "Enclosed_Porch", "Three_season_porch", "Screen_Porch", "Misc_Val"]
    
    data.Garage_Yr_Blt.fillna(data.Year_Built, inplace=True)
    
    #Adding total square footage
    returndata['Total_SF'] = returndata['Total_Bsmt_SF'] + returndata['First_Flr_SF'] + returndata['Second_Flr_SF']
    lam = 0.15
    if settype=='train':
        quantiles={}
        for var in quant_vars:
            q95 = np.quantile(returndata[var],0.95)
            returndata[var][returndata[var]>q95] = q95
            quantiles[var]=q95
        
        returnparams['quants']=quantiles
        
        #numeric_feats = returndata.dtypes[returndata.dtypes != "object"].index
        numeric_feats = returndata.dtypes[returndata.dtypes != "object"].index.tolist()
        numeric_feats.remove('Latitude')
        numeric_feats.remove('Longitude')
        # Check the skew of all numerical features
        skewed_feats = returndata[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
        #print("\nSkew in numerical features: \n")
        skewness = pd.DataFrame({'Skew' :skewed_feats})
        skewness = skewness[abs(skewness) > 0.75]
        #print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))
        skewed_features = skewness.index
        returnparams['skewed_features']=skewed_features
        
        for feat in skewed_features:
            returndata[feat] = boxcox1p(returndata[feat], lam)
    
    if settype=='test':
        for var in quant_vars:
            q95 = params['quants'][var]
            returndata[var][returndata[var]>q95] = q95
            
        skewed_features=params['skewed_features']
        
        for feat in skewed_features:
            returndata[feat] = boxcox1p(returndata[feat], lam)
        
        return returndata
        
    return returnparams, returndata


def main_xgb(write_pred=False):
    train = pd.read_csv('train.csv',index_col='PID')
    test = pd.read_csv('test.csv',index_col='PID')
    
    
    #MSSubClass=The building class
    #alldata['MS_SubClass'] = alldata['MS_SubClass'].apply(str)
    
    train_y = train.Sale_Price
    #test_y = test.Sale_Price
    
    
    
    train = train.drop(['Sale_Price'],axis=1)
    #test = test.drop(['Sale_Price'],axis=1)
    
    train_y = np.log(train_y)
    #test_y = np.log(test_y)
    
    
    preparams, train_x = xgb_prepro(train)
    test_x = xgb_prepro(test,params=preparams,settype='test')
    
    object_cols = list(train_x.dtypes[train_x.dtypes == 'object'].index)
    for col in object_cols:
        le = LabelEncoderExt()
        le.fit(train_x[col])
        train_x[col]=le.transform(train_x[col])
        test_x[col]=le.transform(test_x[col])
    
    model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                      learning_rate=0.05, max_depth=6, 
                      min_child_weight=1.7817, n_estimators=6000,
                      reg_alpha=0.4640, reg_lambda=0.8571,
                      subsample=0.5213, silent=1,
                      random_state =42, nthread = -1)
    
    
    
    model_xgb.fit(train_x, train_y, verbose=False)
    

    y_pred = model_xgb.predict(test_x)
    #score=np.sqrt(np.mean(np.square(y_pred - test_y)))
    returnf = pd.DataFrame(data=np.matrix.transpose(np.array([test_x.index.values.astype(int),np.exp(y_pred)])),  columns=["PID", "Sale_Price"])
    if write_pred:
        #np.savetxt(fname='mysubmission2.txt',X=y_pred)
        returnf.astype({'PID': 'int32'}).to_csv('mysubmission2.txt',index=False)
    #split_score.append(np.sqrt(np.mean(np.square(y_pred - test_y))))
    
    
    return returnf #, score



tic = time.time()


returnf= main_xgb(write_pred=True)
a, l, returnf=main_elastic(write_pred=True)


toc = time.time()
difference = int(toc - tic)

#test_train(2)
#all_test_splits(model='xgbm')
#all_test_splits(model='elastic')




