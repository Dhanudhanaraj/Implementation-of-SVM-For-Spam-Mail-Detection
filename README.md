# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:

To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:

1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Collect a labeled dataset of emails, distinguishing between spam and non-spam.

2.Preprocess the email data by removing unnecessary characters, converting to lowercase, removing stop words, and performing stemming or lemmatization.

3.Extract features from the preprocessed text using techniques like Bag-of-Words or TF-IDF.

4.Split the dataset into a training set and a test set.

5.Train an SVM model using the training set, selecting the appropriate kernel function and hyperparameters.

6.Evaluate the trained model using the test set, considering metrics such as accuracy, precision, recall, and F1 score.

7.Optimize the model's performance by tuning its hyperparameters through techniques like grid search or random search.

8.Deploy the trained and fine-tuned model for real-world use, integrating it into an email server or application.

9.Monitor the model's performance and periodically update it with new data or adjust hyperparameters as needed.

## Program:
```

Program to implement the SVM For Spam Mail Detection..
Developed by:Dhanumalya.D
Register Number:212222230030  

```
```

import chardet
file='/content/spam.csv'
with open(file,'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv('/content/spam.csv',encoding='Windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

```
## Output:
### Result output
![Screenshot from 2023-10-31 22-15-10](https://github.com/Dhanudhanaraj/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119218812/b8ad1ee2-1300-42a1-9e59-51a8540229d7)

### data.head()
![Screenshot from 2023-10-31 22-15-27](https://github.com/Dhanudhanaraj/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119218812/8c476eee-5ee4-4fad-9a5c-f1956aee96cd)

### data.info()
![Screenshot from 2023-10-31 22-15-38](https://github.com/Dhanudhanaraj/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119218812/9d648dd7-da0a-40e2-b7cc-074565a27241)

### data.isnull().sum()
![Screenshot from 2023-10-31 22-15-46](https://github.com/Dhanudhanaraj/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119218812/b30baf73-fdf2-403d-96ce-71728ee8c806)

### Y_prediction value
![Screenshot from 2023-10-31 22-15-58](https://github.com/Dhanudhanaraj/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119218812/fb7a6d41-592b-4f30-9da7-a80b6d8c98de)

### Accuracy value
![Screenshot from 2023-10-31 22-16-06](https://github.com/Dhanudhanaraj/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119218812/4c2f4cdf-aef8-4f79-89e5-f5eb7d0b6f93)

## Result:

Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
