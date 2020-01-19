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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def cm_analysis(y_true, y_pred, labels, ymap=None, figsize=(10,10)):
    if ymap is not None:
        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]
        labels = [ymap[yi] for yi in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
              
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    ##CUSTOMIZE GRAFTS - BORDERS THICKNESS 0,75 N COLOUR - BLACK
    sns.heatmap(cm, annot=annot, fmt='', ax=ax, cmap="YlGnBu", linewidths=0.75, linecolor='black')
    
    plt.title("Random Forest 100")
    plt.savefig("Random Forest")
    
    plt.show()
#============================================================================================================
#|||||||||||||||||||||||||||||||||| IMPORT MODEL ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#============================================================================================================ 


from sklearn.model_selection import train_test_split
root ="C:/Users/Another/CNN/Batch Trainning/Dataset/"

datafile = root +"data"+"NewTrue_Unique-Out-of-SampleMel-Log-TT-7258 A-18 H-36 C-18 V-27-98-40 15-Dec-2019-1044H.npy"
targetfile = root +"target"+"NewTrue_Unique-Out-of-SampleMel-Log-TT-7258 A-18 H-36 C-18 V-27-98-40 15-Dec-2019-1044H.npy"

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


y_out_ONEHOT = OneHotEncode(OutloadTarget,num_classes)


y_out_ONEHOT.shape


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


XX = flattenFeature(XX)
print('Testing Features Shape:', XX.shape)

#============================================================================================================
#|||||||||||||||||||||||||||||||||| IMPORT MODEL ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#============================================================================================================ 


X_train_out, X_test_out, y_train_out, y_test_out = train_test_split(XX, yy, test_size=0.50, random_state=42, stratify = yy)

print(len(X_train_out))


import pickle
# save the model to disk
filename = 'RF_Local.sav'
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
y_pred=loaded_model.predict(X_test_out)

cm_analysis(y_test_out, y_pred, [0,1,2,3], ymap=None, figsize=(8,8))



