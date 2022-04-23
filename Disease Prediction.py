#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:





# In[2]:


data = pd.read_csv('Training.csv').dropna(axis = 1)


# # Exploring
# 

# In[3]:


disease_counts = data["prognosis"].value_counts()
temp_df = pd.DataFrame({
    "Disease": disease_counts.index,
    "Counts": disease_counts.values
})


# In[ ]:





# In[5]:


data.head()


# In[6]:



disease_counts = data["prognosis"].value_counts()
prog = pd.DataFrame({
    "Disease": disease_counts.index,
    "Counts": disease_counts.values
})


# In[8]:


plt.figure(figsize = (15,8))
sns.barplot(x = "Disease", y = "Counts", data = prog,palette="magma")
plt.xticks(rotation=90)
plt.show()


# In[ ]:





# # Encoding data to train 

# In[9]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
data["prognosis"] = encoder.fit_transform(data["prognosis"])


# # Spliting the data to train and test

# In[10]:


from sklearn.model_selection import train_test_split, cross_val_score

X = data.iloc[:,:-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test =train_test_split(
  X, y, test_size = 0.2, random_state = 24)
 
print(f"Train: {X_train.shape}, {y_train.shape}")
print(f"Test: {X_test.shape}, {y_test.shape}")


# In[17]:


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix


# In[18]:


def cv_scoring(estimator, X, y):
    return accuracy_score(y, estimator.predict(X))


# In[19]:


model = GaussianNB()


# In[99]:





# In[20]:


scores = cross_val_score(model, X, y, cv = 10,
                             n_jobs = -1,
                             scoring = cv_scoring)
print(f"Scores: {scores}")
print(f"Mean Score: {np.mean(scores)}")


# In[21]:


model.fit(X_train, y_train)
preds = model.predict(X_test)
print(f"Accuracy on train data by Gaussian Naive Bayes Classifier: {accuracy_score(y_train, model.predict(X_train))*100}")
 
print(f"Accuracy on test data by Gaussian Naive Bayes Classifier: {accuracy_score(y_test, preds)*100}")
cf_matrix = confusion_matrix(y_test, preds)
plt.figure(figsize=(12,8))
sns.heatmap(cf_matrix, annot=True)
plt.title("Confusion Matrix for Gaussian Naive Bayes Classifier on Test Data")
plt.show()


# # Train the whole data 

# In[22]:


final_cl = GaussianNB()
final_cl.fit(X,y)


# In[23]:


test_data = pd.read_csv("Testing.csv").dropna(axis=1)
 
test_X = test_data.iloc[:, :-1]
test_Y = encoder.transform(test_data.iloc[:, -1])


# # Making Predection
# 

# In[24]:


pred = final_cl.predict(test_X)


# In[28]:


accuracy_score(test_Y, pred)*100


# In[29]:


cf_matrix = confusion_matrix(test_Y,pred)
plt.figure(figsize=(12,8))
 
sns.heatmap(cf_matrix, annot = True)
plt.title("Confusion Matrix for Gauson naive bayes Model on Test Dataset")
plt.show()


# # Function to take symptoms and predict the condition 

# In[58]:


symptoms = X.columns.values

symptom_index = {}
for index, value in enumerate(symptoms):
    symptom = " ".join([i.capitalize() for i in value.split("_")])
    symptom_index[symptom] = index
 
data_dict = {
    "symptom_index":symptom_index,
    "predictions_classes":encoder.classes_}
 


# In[59]:


def predictDisease(symptoms):
    symptoms = symptoms.split(",")
    input_data = [0] * len(data_dict["symptom_index"])
    for symptom in symptoms:
        index = data_dict["symptom_index"][symptom]
        input_data[index] = 1
    input_data = np.array(input_data).reshape(1,-1)
        
    nb_prediction = data_dict["predictions_classes"][final_cl.predict(input_data)[0]]
    
    predictions = {
        "Predicted Condition": nb_prediction,
    }
    return predictions


# In[60]:


print(predictDisease("Yellow Crust Ooze,Red Sore Around Nose"))


# In[ ]:





# In[ ]:




