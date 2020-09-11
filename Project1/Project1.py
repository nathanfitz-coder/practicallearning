import pandas as pd
import xgboost as xgb

data = pd.read_csv('Ames_data.csv')


testIDs = pd.read_table("project1_testIDs.dat",header=None,sep = ' ').values
j = 2

testIDs[:,j]


testidx = data.index.isin(testIDs[:,j])
train = data[~testidx]
test = data[testidx]



train_x = train.iloc[:,0:82]
train_y = train.iloc[:,82]

test_x = test.iloc[:,0:82]
test_y = test.iloc[:,82]


pd.write_csv(train,"train.csv",row.names=FALSE)
pd.write_csv(test, "test.csv",row.names=FALSE)
pd.write_csv(test.y,"test_y.csv",row.names=FALSE)


