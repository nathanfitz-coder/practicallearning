import pandas as pd

data = pd.read_csv('Ames_data.csv')
testIDs = pd.read_table("project1_testIDs.dat",header=None,sep = ' ').values


j = 2


testidx = data.index.isin(testIDs[:,j])
train = data[~testidx]
test = data[testidx]


train.to_csv("train.csv",index=False)
test.to_csv("test.csv",index=False)


