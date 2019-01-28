#%matplotlib inline
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
from adaline import AdalineGD

if __name__=='__main__':
	df = pd.read_csv('iris.data.csv', header=None)
	y = df.iloc[0:100, 4].values
	y = np.where(y=='Iris-setosa',-1,1)
	X = df.iloc[0:100, [0,2]].values
	
	# standarization
	# (x- 평균)/sd
	X_std = np.copy(X)
	X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:, 0].std()
	X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:, 1].std()
	
	adal = AdalineGD(eta=0.01, n_iter = 15).fit(X_std,y)
	plt.plot(range(1, len(adal.cost_) +1 ), adal.cost_, marker='o')
	plt.xlabel('Epochs')
	plt.ylabel('SSE')
	plt.title('Adaline standardized - Learning rate 0.01')
	plt.show()