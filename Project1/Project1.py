import pandas as pd

data = pd.read_csv('Ames_data.csv')


testIDs = pd.read_csv("project1_testIDs.dat")
j <- 2
train <- data[-testIDs[,j], ]
test <- data[testIDs[,j], ]
test.y <- test[, c(1, 83)]
test <- test[, -83]
write.csv(train,"train.csv",row.names=FALSE)
write.csv(test, "test.csv",row.names=FALSE)
write.csv(test.y,"test_y.csv",row.names=FALSE)