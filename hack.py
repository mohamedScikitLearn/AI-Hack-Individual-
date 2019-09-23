import pandas as pd
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

train=pd.read_csv('fr/train/invoice_train.csv')
train_inv=pd.read_csv('fr/train/client_train.csv')




test_inv=pd.read_csv('fr/test/client_test.csv')
test=pd.read_csv('fr/test/invoice_test.csv')

X_test=test.drop(['invoice_date'],axis=1)
X_test=X_test.drop(['client_id'],axis=1)


def counter_type(x):
    if x=="ELEC":     
        return 1
    else:
        return 0
    
############################################################
training_set= pd.concat([train_inv, train], axis=1, join='inner')
testing_set= pd.concat([test_inv, test], axis=1, join='inner')
############################################################



training_set['counter_type']=training_set['counter_type'].apply(lambda x:counter_type(x))
testing_set['counter_type']=testing_set['counter_type'].apply(lambda x:counter_type(x))

####### DROPING 

#test_set=test_set.drop(['client_id','invoice_date','old_index',
#                      'new_index','creation_date',
#                      'months_number'],axis=1)

    
Y = training_set["target"].values


training_set=training_set.drop(['client_id','invoice_date',
                      'creation_date',
                      'target'],axis=1)
    
testing_set=testing_set.drop(['client_id','invoice_date','creation_date'
                      ],axis=1)

#y=train_inv.iloc[:,5:]

#train['invoice_date'] = pd.to_datetime(train['invoice_date'])
#
#
#import time
#import datetime
#  = "01/12/2011"
#time.mktime(datetime.datetime.strptime(s, "%d/%m/%Y").timetuple())
#s=train['invoice_date'].values

trainii=training_set.values

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=630, max_depth=30,
                             random_state=100)
clf.fit(training_set, Y)  


 
### helper code
for i in training_set.columns:
    print(i,training_set[i].unique())
    print("%%%%%%%%")
#
 

#y_pred=clf.predict(testing_set)

probs = clf.predict_proba(testing_set)
# keep probabilities for the positive outcome only
probs = probs[:, 1]

#probs

#########

# create submission DataFrame
submission = pd.DataFrame({'client_id':test_inv['client_id'],'target':probs})

#Visualize the first 5 rows
submission.head()

filename = '8-th-BERRIMI_Mohamed-Sub.csv'

submission.to_csv(filename,index=False)

print('Saved file: ' + filename)
############ Feature scaling 
from sklearn.preprocessing import StandardScaler 

scaler = StandardScaler() 
from sklearn import preprocessing

normalized_X = preprocessing.normalize(training_set)
normalized_X_test = preprocessing.normalize(testing_set)



##### Fit scaled data to RF classifier

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=30, max_depth=8,
                             random_state=0)
clf.fit(normalized_X, Y)  

probs = clf.predict_proba(testing_set)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# create submission DataFrame
submission = pd.DataFrame({'client_id':test_inv['client_id'],'target':probs})

#Visualize the first 5 rows
submission.head()

filename = '2nd-BERRIMI_Mohamed-Sub.csv'

submission.to_csv(filename,index=False)

print('Saved file: ' + filename)

############ XGBOOST 


from xgboost import XGBClassifier

model = XGBClassifier()
model.fit(normalized_X, Y)

probs = clf.predict_proba(testing_set)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# create submission DataFrame
submission = pd.DataFrame({'client_id':test_inv['client_id'],'target':probs})

#Visualize the first 5 rows
submission.head()

filename = '3rd-BERRIMI_Mohamed-Sub.csv'

submission.to_csv(filename,index=False)

print('Saved file: ' + filename)
##########"


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
Train_set = sc.fit_transform(training_set)
Test_SET = sc.transform(testing_set)
 
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=300, max_depth=2,
                             random_state=0)
clf.fit(X_train, Y)  

probs = clf.predict_proba(X_test)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# create submission DataFrame
submission = pd.DataFrame({'client_id':test_inv['client_id'],'target':probs})

#Visualize the first 5 rows
submission.head()

filename = '314-BERRIMI_Mohamed-Sub.csv'

submission.to_csv(filename,index=False)

print('Saved file: ' + filename)


######## Deep learning 


import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation, BatchNormalization, regularizers
from keras.layers.noise import GaussianNoise
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils.np_utils import to_categorical

y_trainHot = to_categorical(Y, num_classes = 2)
 



probs = classifier.predict_proba(testing_set)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# create submission DataFrame
submission = pd.DataFrame({'client_id':test_inv['client_id'],'target':probs})

#Visualize the first 5 rows
submission.head()

filename = '6th-BERRIMI_Mohamed-Sub.csv'

submission.to_csv(filename,index=False)

print('Saved file: ' + filename)




from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(training_set)
X_test = sc.transform(testing_set)
