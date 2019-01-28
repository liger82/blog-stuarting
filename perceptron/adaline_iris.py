#%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style
from adaline import AdalineGD

if __name__=='__main__':
	df = pd.read_csv('iris.data.csv', header = None)
	# print(df.iloc[0:2,4].values)
	# 5번째(==인덱스 4) 열이 label임. 그 값을 가져오는 것
	y = df.iloc[0:100, 4].values
	# 세토사이면 0, 아니면 1
	y = np.where(y=='Iris-setosa',0,1)
	# feature는 두 개만 사용함.(총 4개 있음)
	X = df.iloc[0:100,[0,2]].values
	
	fig, ax = plt.subplots(nrows =1, ncols=2, figsize=(8,4))
	print("learning rate가 클 때 : 발산해버림")
	adal = AdalineGD(eta=0.01,n_iter=10).fit(X,y)
	ax[0].plot(range(1,len(adal.cost_) + 1), np.log10(adal.cost_), marker='o')
	ax[0].set_xlabel('Epochs')
	ax[0].set_ylabel('log(SQE)')
	ax[0].set_title('Adaline - Learning rate 0.01')
	
	print("\nlearning rate가 작을 때")
	adal2 = AdalineGD(eta =0.0001,n_iter=10).fit(X,y)
	ax[1].plot(range(1, len(adal2.cost_)+1), np.log10(adal2.cost_), marker='o')
	ax[1].set_xlabel('Epochs')
	ax[1].set_ylabel('log(SQE)')
	ax[1].set_title('Adaline - Learning rate 0.0001')
	
	plt.subplots_adjust(left=0.1, bottom=0.2, right=0.95, top=0.9, wspace=0.5, hspace=0.5)
	plt.show()
	