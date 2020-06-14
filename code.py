#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler                  #library to scale the given data
from sklearn.model_selection import train_test_split            #for even splitting of test and train data
import matplotlib.pyplot as plt                                 #for graph plotting
from keras.layers import Dropout, Dense, LSTM, Embedding, Conv1D, MaxPool1D, Flatten
from keras import regularizers
from keras.optimizers import Adam                              #to generate neural network model
from keras.models import Sequential                            #and add functions at each layer
import keras                                                    
from sklearn.decomposition import PCA                           #for better fit of prediction


# In[15]:


Columns=["Customer_ID","rev_gen", "change_mou", "change_rev", "comp_vce_Mean", "comp_dat_Mean", "actvsubs","avgrev", "avg3rev", "avg3mou", "avg6mou",
         "avg6rev", "adults", "income","lor", "creditcd"]
#features required for the prediction of the generated revnue


# In[16]:


df=pd.read_csv("Telecom_customer churn3.csv")                   #reads the data stored in csv format and stores it in dataframe df
df = df[Columns].set_index('Customer_ID')                       #customer_ID set as index as it is unique for each of the rows 
                                                                #and only the required columns which is mentioned in Columns is extracted


# In[17]:


df.describe()                                                   #description of dataset going to be used


# In[18]:


#cleaning of data

df.adults.fillna('2.00', inplace=True)                                        #empty elements of low priority data is filled 
df.income.fillna('4.40', inplace=True)                                        #with average value of that column present  
df.lor.fillna('7.00', inplace=True)                                           #throughout the dataset
df.drop(df[ (df.rev_gen > 186.30) ].index, axis=0, inplace=True)
df.drop(df[ (df.change_mou > 1000) | (df.change_mou < -1000) ].index, axis=0, inplace=True)
df.drop(df[ (df.comp_vce_Mean > 284) ].index, axis=0, inplace=True)
df.drop(df[ (df.avgrev > 139.05) ].index, axis=0, inplace=True)             #rows having certain very high or low from the mean
df.drop(df[ (df.avg3mou > 1543.05) ].index, axis=0, inplace=True)           #value is removed
df.drop(df[ (df.avg6mou > 1443.40) ].index, axis=0, inplace=True)
df.drop(df[ (df.avg6rev > 130.05) ].index, axis=0, inplace=True)
df=df.dropna()                                                              #rows still having missing elements is removed
one_hot = pd.get_dummies(df['creditcd'])                                    #one hot encoding of column creditcd
df = df.drop('creditcd',axis = 1)                                           
df=df.join(one_hot)
df                                                                          #final dataset going to be used for training


# In[19]:


y = df['rev_gen'].values                                                             #output column
X = df
X.drop(['rev_gen'],axis = 1, inplace=True)                                           #input features' dataframe
sc = MinMaxScaler()
sc_x = sc.fit(X)
X = sc_x.transform(X)                                                                #scaling of input features
sc = MinMaxScaler()
sc_y = sc.fit(y.reshape(-1,1))
y = sc_y.transform(y.reshape(-1,1))                                                  # scaling of output data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.1, random_state=21) # splitting randomly training set(90%) 
                                                                                           #and testing set(10%) of data


# In[20]:


pca = PCA(n_components = 15)
X_train = pca.fit_transform(X_train)                       #pca transformation of training and test data for better fit of model
X_test = pca.transform(X_test)


# In[21]:


#definition of neural network model
def ann_model():                                                     
    model = Sequential()                                             
    model.add(Dense(15, activation="relu", input_dim=15))            
    model.add(Dropout(0.1))
    model.add(Dense(activation="sigmoid", units=256))
    model.add(Dropout(0.1))
    model.add(Dense(activation="sigmoid", units=256))
    model.add(Dropout(0.1))
    model.add(Dense(activation="sigmoid", units=256))
    model.add(Dropout(0.1))
    model.add(Dense(activation="sigmoid", units=128))
    model.add(Dropout(0.1))
    model.add(Dense(activation="sigmoid", units=128))
    model.add(Dropout(0.1))
    model.add(Dense(activation="sigmoid", units=128))
    model.add(Dropout(0.1))
    model.add(Dense(activation="sigmoid", units=64))
    model.add(Dropout(0.3))
    model.add(Dense(activation="sigmoid", units=1))
    model.compile(optimizer = 'Adam',loss = 'mse', metrics = [ 'mae'])    #Adam optimiser used for optimisation of model
    return model                                                          #loss reported in mean squared error and 
                                                                          #mean absolute error used as the metric
model = ann_model()
model.summary()                                      #model summary


# In[22]:


#training on the data
try:
    tbCallbacks = keras.callbacks.TensorBoard(log_dir='./logs_3', histogram_freq=0, batch_size=128, write_graph=True, write_grads=True, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')
    model.fit(X_train, y_train, callbacks=[tbCallbacks],epochs=10,batch_size=32)
except KeyboardInterrupt:
    print("\nInterrupting")


# In[12]:


#prediction on the test data 
y_pred = model.predict(X_test)
y_pred = sc_y.inverse_transform(y_pred)
y_t = sc_y.inverse_transform(y_test)
y_subset=y_pred[100:200]
y_sub=y_t[100:200]


# In[13]:


#plot of predicted revnue with blue and given value with orange on the y-axis and customer id on the x-axis.
plt.figure(figsize=(25,15))
plt.plot(y_subset, label ='y_pred')
plt.plot(y_sub, label = 'real')
plt.xlabel('features_row number')
plt.ylabel('revenue')
plt.title("using ann")
plt.legend()
plt.savefig('reportann1.png')


# In[ ]:





# In[ ]:




