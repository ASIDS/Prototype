import platform; print(platform.platform())
import sys; print("Python", sys.version)
import numpy as np; print("NumPy", np.__version__)
import scipy; print("SciPy", scipy.__version__)
import sklearn; print("Scikit-Learn", sklearn.__version__)


def OneHotEncode(x,classAmount):
    import numpy as np
    from numpy import argmax
    # integer encode input data
    integer_encoded = list(np.floor(x).astype(int))
    # one hot encode
    onehot_encoded = list()
    for value in integer_encoded:
        temp = [0 for _ in range(classAmount)]
        temp[value] = 1
        onehot_encoded.append(temp)
    return np.array(onehot_encoded)
# invert encoding

#============================================================================================================
#|||||||||||||||||||||||||||||||||| IMPORT MODEL ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#============================================================================================================ 
root =""

datafile = root +"data"+"DSTrain.npy"
targetfile = root +"target"+"DSTrain.npy"

InsampleloadTarget = np.load(targetfile) # load
InsampleloadData = np.load(datafile) #

X=InsampleloadData
y=InsampleloadTarget


AAA = list(y).count(0.000)
HHH = list(y).count(1.000)
CCC = list(y).count(2.000)
VVV = list(y).count(3.000)
i=AAA+HHH+CCC+VVV
Aper = int(AAA/i*100)
Hper = int(HHH/i*100)
Cper = int(CCC/i*100)
Vper = int(VVV/i*100)
print(i,"Data")
print(AAA,"samples 0-Ambience ",Aper,"%")
print(HHH,"samples 1-Hatchet ",Hper,"%")
print(CCC,"samples 2-Chainsaw ",Cper,"%")
print(VVV,"samples 3-Vehicle ",Vper,"%")
print ("Final Data Extracted Shape",X.shape)
print ("Final Target Extracted Shape",y.shape)

from sklearn.model_selection import train_test_split
root =""

datafile = root +"data"+"DSTest.npy"
targetfile = root +"target"+"DSTest.npy"


OutloadTarget = np.load(targetfile) # load
OutloadData = np.load(datafile) #
X_out = OutloadData
y_out = OutloadTarget

XX = OutloadData
yy = OutloadTarget

AAA = list(yy).count(0.000)
HHH = list(yy).count(1.000)
CCC = list(yy).count(2.000)
VVV = list(yy).count(3.000)
i=AAA+HHH+CCC+VVV
Aper = int(AAA/i*100)
Hper = int(HHH/i*100)
Cper = int(CCC/i*100)
Vper = int(VVV/i*100)
print(i,"Data")
print(AAA,"samples 0-Ambience ",Aper,"%")
print(HHH,"samples 1-Hatchet ",Hper,"%")
print(CCC,"samples 2-Chainsaw ",Cper,"%")
print(VVV,"samples 3-Vehicle ",Vper,"%")
print ("Final Data Extracted Shape",XX.shape)
print ("Final Target Extracted Shape",yy.shape)


num_classes = 4

y_in_ONEHOT = OneHotEncode(InsampleloadTarget,num_classes)
y_out_ONEHOT = OneHotEncode(OutloadTarget,num_classes)

y_in_ONEHOT.shape
y_out_ONEHOT.shape

print('Training Features Shape:', InsampleloadData.shape)
print('Training Labels Shape:', y_in_ONEHOT.shape)
print('Testing Features Shape:', OutloadData.shape)
print('Testing Labels Shape:', y_out_ONEHOT.shape)

#============================================================================================================
#|||||||||||||||||||||||||||||||||| IMPORT MODEL ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#============================================================================================================ 

def flattenFeature(x):
    flats = []
    print(len(x))
    for i in range(len(x)):
        flat = np.ndarray.flatten(x[i,:,:])
        flats.append(flat)
    return np.array(flats)







#============================================================================================================
#|||||||||||||||||||||||||||||||||| IMPORT MODEL ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#============================================================================================================ 

X_train = X
y_train = y
print(len(X_train))

X_test_out= XX
y_test_out = yy
print(len(X_test_out))



#============================================================================================================
#|||||||||||||||||||||||||||||||||| IMPORT MODEL ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#============================================================================================================ 
#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=25, 
                               bootstrap = True,
                               max_features = 'sqrt',random_state=42)

clf.fit(X_train,y_train)

#============================================================================================================
#|||||||||||||||||||||||||||||||||| IMPORT MODEL ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#============================================================================================================ 
from sklearn import metrics

y_pred=clf.predict(X_test_out)
# Model Accuracy, how often is the classifier correct?


print("Accuracy:",metrics.accuracy_score(y_pred, y_test_out))

import pickle
# save the model to disk
filename = 'RF_Local.sav'
pickle.dump(clf, open(filename, 'wb'))
print("Model Created : ", filename)

