
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.decomposition import PCA,KernelPCA

from sklearn.metrics import fbeta_score
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.linear_model import LinearRegression as LR
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import balanced_accuracy_score as bas
from sklearn.metrics import plot_confusion_matrix as pcm
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn import tree
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.dummy import DummyClassifier

from sklearn import linear_model
import sweetviz as sv


from sklearn.ensemble import HistGradientBoostingRegressor

def perform_method_null(modell,train_x,train_y,test_x):
	print("Forgot to select method...exiting")
	exit()



models = [
	DummyClassifier(),
	KNeighborsClassifier(3),
	
	GradientBoostingClassifier(n_estimators=1, learning_rate=1.0,max_depth=3, random_state=0),
	
	
	#SVC(kernel="linear", C=0.025),  #takes forever
	#SVC(gamma=2, C=1),  #takes forever
	#GaussianProcessClassifier(1.0 * RBF(1.0)),  #takes forever
	
	DTC(max_depth=5),
	RandomForestClassifier(max_depth=2, n_estimators=60, max_features=2),
	MLPClassifier(alpha=1, max_iter=1000),
	AdaBoostClassifier(),
	GaussianNB(),
	QuadraticDiscriminantAnalysis(),
	]

#modell = BaseEstimator()
#method = perform_method_null
train_x = None
train_y = None
test_x = None
test_y = None


# Helper functions

def export(Y,model,method):
	filename = model.__class__.__name__ +"_"+ method.__name__ + ".csv"
	print("Exporting to","./export/"+filename)
	f = open(filename,'w')

	f.write("Id,Predicted\n")
	for i,val in enumerate(Y):
		f.write("{},{}\n".format(i,val))

	f.flush()
	f.close()

def perform_method_null(modell,train_x,train_y,test_x):
	print("Forgot to select method...exiting")
	exit()

def equalize_data(x,y):
	data = pd.concat([x,y.Expected],1)# adding result here remove later step
	data = data.reset_index(drop=True)
	#Test = pd.read_csv('data/X_test.csv')
	#y = pd.read_csv('data/y.csv')

	class_1 = np.where(y.Expected == 1)
	class_0 = np.where(y.Expected == 0)

	class_1 = pd.DataFrame(data,index=class_1[0], columns=data.columns)
	class_0 = pd.DataFrame(data,index=class_0[0], columns=data.columns)


	class_0 = class_0.sample(n = len(class_1))

	frames = [class_1, class_0]

	result = pd.concat(frames)

	x = result.sample(n = len(result))#randomise
	y = pd.DataFrame(x['Expected'])
	x.drop('Expected', inplace=True, axis=1) # we delete the result
	return x,y

def load_data(isdeploy = False):
	print("Loading data..")
	train_x = pd.read_csv('../data/X_train.csv')
	train_y = pd.read_csv('../data/y_train.csv')
	train_x.drop('Id', inplace=True, axis=1) # staticly drop the id collumn
	train_y.drop('Id', inplace=True, axis=1) # staticly drop the id collumn
	if isdeploy:
		test_x = pd.read_csv('../data/X_test.csv')
		test_x.drop('Id', inplace=True, axis=1) # staticly drop the id collumn
		return train_x, test_x, train_y,None # here there are no test_y
	else:
		return train_test_split(train_x, train_y,test_size=0.3, random_state=42)

def SeparateValuesHists(data):
	visited = []
	values = []
	hists = []
	for col in data.columns:
		simbol = col[0:2]
		if simbol in visited:
			continue
		visited.append(simbol)
		#print(simbol)
		sim_cols = []
		#get all similar colum
		for col2 in data.columns:
			if simbol in col2:
				sim_cols.append(col2)
		if len(sim_cols) > 1:
			hists.append(sim_cols)
		else:
			values.append(sim_cols[0])
	return values,hists


def HistNormalize(data):
	print("HistNormalize")
	visited = []

	for col in data.columns:
		simbol = col[0:2]
		if simbol in visited:
			continue
		visited.append(simbol)
		#print(simbol)
		sim_cols = []
		#get all similar colum
		for col2 in data.columns:
			if simbol in col2:
				sim_cols.append(col2)
		
		if len(sim_cols) > 1:
			selected_block = data[sim_cols] 
			selected_block = selected_block.div(selected_block.sum(axis=1),axis = 0)
			data[sim_cols] = selected_block 

	return data 


##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################

#Method functions
def method_allfeature_no_optim(modell,train_x,test_x,train_y):
	print("Performing ", modell.__class__.__name__ +"_"+ method.__name__ + "...")
	
	train_x = train_x.fillna(0)
	test_x = test_x.fillna(0)

	
	modell.fit(train_x,train_y)

	pcm(model,test_x,test_y)
	plt.show()
	
	Y = modell.predict(test_x)

	return Y


def method_normhist_allvalue_zeronan(modell,train_x,test_x,train_y):
	print("Performing ", modell.__class__.__name__ +"_"+ method.__name__ + "...")
	
	
	train_x = train_x.fillna(0)
	test_x = test_x.fillna(0)

	train_x = HistNormalize(train_x)
	test_x = HistNormalize(test_x)

	train_x = train_x.fillna(0) # we do it again due to 0 division
	test_x = test_x.fillna(0)

	values,hists_vec = SeparateValuesHists(train_x)
	hists = []
	for h in hists_vec:
		hists.extend(h)
	
	modell.fit(train_x,train_y)
	
	Y = modell.predict(test_x)

	return Y



def method_normhist_allvalue_minusnan(modell,train_x,test_x,train_y):
	print("Performing ", modell.__class__.__name__ +"_"+ method.__name__ + "...")
	
	
	train_x = train_x.fillna(-1)
	test_x = test_x.fillna(-1)

	train_x = HistNormalize(train_x)
	test_x = HistNormalize(test_x)

	train_x = train_x.fillna(-1) 
	test_x = test_x.fillna(-1)

	values,hists_vec = SeparateValuesHists(train_x)
	hists = []
	for h in hists_vec:
		hists.extend(h)
	
	modell.fit(train_x,train_y)
	
	Y = modell.predict(test_x)

	return Y


def method_normhist_only_minusnan(modell,train_x,test_x,train_y):
	print("Performing ", modell.__class__.__name__ +"_"+ method.__name__ + "...")
	train_x = train_x.fillna(-1)
	test_x = test_x.fillna(-1)

	train_x = HistNormalize(train_x)
	test_x = HistNormalize(test_x)

	train_x = train_x.fillna(-1) # we do it again do to 0 division
	test_x = test_x.fillna(-1)

	values,hists_vec = SeparateValuesHists(train_x)
	hists = []
	for h in hists_vec:
		hists.extend(h)
	

	#select only histograms
	train_x = train_x[hists]
	test_x = test_x[hists]
	

	
	modell.fit(train_x,train_y)
	
	Y = modell.predict(test_x)

	return Y


def method_values_only_minusnan(modell,train_x,test_x,train_y):
	print("Performing ", modell.__class__.__name__ +"_"+ method.__name__ + "...")
	train_x = train_x.fillna(-1)
	test_x = test_x.fillna(-1)
	train_x = HistNormalize(train_x)
	test_x = HistNormalize(test_x)
	train_x = train_x.fillna(-1) # we do it again do to 0 division
	test_x = test_x.fillna(-1)
	values,hists_vec = SeparateValuesHists(train_x)
	hists = []
	for h in hists_vec:
		hists.extend(h)
	#select only histograms
	train_x = train_x[values]
	test_x = test_x[values]
	modell.fit(train_x,train_y)
	Y = modell.predict(test_x)
	return Y


def method_no_histogram_pca_70(modell,train_x,test_x,train_y):
	print("Performing ", modell.__class__.__name__ +"_"+ method.__name__ + "...")
	values,_ = SeparateValuesHists(train_x)
	train_x = train_x[values]
	test_x = test_x[values]
	train_x = train_x.fillna(-1)
	test_x = test_x.fillna(-1)
	pca = PCA(n_components=70)
	pca = pca.fit(train_x)
	train_x = pca.transform(train_x)
	test_x = pca.transform(test_x)
	print("Training..")
	modell.fit(train_x,train_y)
	print("Predicting..")
	Y = modell.predict(test_x)

	return Y

def method_visualize_normalized_pca_histograms(modell,train_x,test_x,train_y):
	#print("Performing ", modell.__class__.__name__ +"_"+ method.__name__ + "...")
	
	print("Preprocess..")
	train_x = train_x.fillna(-1)
	test_x = test_x.fillna(-1)

	train_x = HistNormalize(train_x)
	test_x = HistNormalize(test_x)

	train_x = train_x.fillna(-1) # we do it again do to 0 division
	test_x = test_x.fillna(-1)

	values,hists = SeparateValuesHists(train_x)

	
	global test_y
	for h in hists:
		#select only histograms
		X = train_x[h]
		TX = test_x[h]

		pca = PCA(n_components=9)
		pca = pca.fit(X)
		X = pca.transform(X)
		TX = pca.transform(TX)
		
		fig = plt.figure()
		ax = fig.add_subplot(1,2,1,projection='3d')
		ax.scatter(X [:,0],X [:,1],X [:,2],c=train_y.Expected,s = 40)
		ax = fig.add_subplot(1,2,2,projection='3d')
		ax.scatter(X [:,3],X [:,4],X [:,5],c=train_y.Expected)
		plt.show()
		

	print("Training..")
	modell.fit(train_x,train_y)
	print("Predicting..")
	Y = modell.predict(test_x)

	return Y


methods = [
	#method_allfeature_no_optim,
	#method_normhist_allvalue_zeronan,
	#method_normhist_allvalue_minusnan,
	#method_normhist_only_minusnan,
	#method_values_only_minusnan,
	#method_no_histogram_pca_70,
]


train_x, test_x, train_y, test_y = load_data(False)
train_x,train_y = equalize_data(train_x,train_y)
method_visualize_normalized_pca_histograms(None,train_x, test_x, train_y)

quit()


print("All modell testing....")

f = open("performance.csv",'w')


f.write("Model,Method,F3,acc,error,recall,precision\n")
for model in models:
	for method in methods:
		name = model.__class__.__name__ + "_" + method.__name__
		print(name)
		
		train_x, test_x, train_y, test_y = load_data(False)
		#train_x,train_y = equalize_data(train_x,train_y)
		if model is object:
			raise Exception("Forgot to select a model..")
		Y = method(model,train_x, test_x, train_y)
		
		#export(Y,model,method)
		#exit()
		error = 0
		for i in range(len(test_y)):
			if test_y._values[i] != Y[i]:
				#print(test_y._values[i],Y[i],Y_test_proba[i])
				error += 1 
		#print("Error count =",error,"/",len(test_y))
		Acc = acc(test_y, Y)
		F3 = fbeta_score(test_y, Y, average='binary', beta=3)
		print("F3,",F3)
		recall = recall_score(test_y,Y)
		precision = precision_score(test_y,Y)
		#print("precision,",precision)
		#print("recall,",recall)
		#print("f_measure",f_measure(recall,precision))
		print("-----------------------------------")
		f.write("{},{},{},{},{},{},{}\n".format( model.__class__.__name__ , method.__name__,F3,Acc,error,recall,precision))
	#	break
	#break

f.flush()
f.close()












