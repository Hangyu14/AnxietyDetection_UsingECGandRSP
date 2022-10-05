from importlib.resources import path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
def process(path):
    boxdata = pd.read_csv(path)
    boxdata.drop(columns=boxdata.columns[0], axis=1, inplace=True)
    print(boxdata)
    return boxdata


# generate Box-plot for each classifer

data1 = process('new/boxplot_data1.csv')
data2 = process('new/boxplot_data2.csv')
data3 = process('new/boxplot_data3.csv')

svm_1=data1.iloc[0]
knn_1=data1.iloc[1]
mlp_1=data1.iloc[2]

svm_2=data2.iloc[0]
knn_2=data2.iloc[1]
mlp_2=data2.iloc[2]

svm_3=data3.iloc[0]
knn_3=data3.iloc[1]
mlp_3=data3.iloc[2]

def result(data):
    print(np.mean(data))
    print(np.std(data))

result(svm_1)
result(knn_1)
result(mlp_1)
result(svm_2)
result(knn_2)
result(mlp_2)
result(svm_3)
result(knn_3)
result(mlp_3)