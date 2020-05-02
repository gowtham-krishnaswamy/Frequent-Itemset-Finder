# Apriori

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Preprocessing
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)

# Dimensions of data
print("Dimensions: ",dataset.shape)


transactions = []
for i in range(0, dataset.shape[0]):
    transactions.append([str(dataset.values[i,j]) for j in range(0, dataset.shape[1])  if dataset.values[i,j] is not np.nan])

# Training Apriori on the dataset
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

# Visualising the results
results = list(rules)

#Write rules to a txt file
o=open("ans.txt","w")

for item in results:
    pair = item[0] 
    items = [x for x in pair]
    o.writelines(["Rule      : ",str(items[0]), " -> ", items[1],"\n"])
    o.writelines(["Support   : " , str(item[1]),"\n"])
    o.writelines(["Confidence: " , str(item[2][0][2]),"\n"])
    o.writelines(["Lift      : " + str(item[2][0][3]),"\n"])
    o.writelines(["=================================================\n"])

o.close()