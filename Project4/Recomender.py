import pandas as pd


user_header = ["UserID","Gender","Age","Occupation","Zip-code"]
users = pd.read_csv('MovieData/users.dat',sep='::',names=user_header)




