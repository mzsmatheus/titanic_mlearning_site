#imports de ciencia de dados e machine learning
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

#imports dos classificadores
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

def treinamentoteste():

    url = 'https://learnenough.s3.amazonaws.com/titanic.csv'
    titanic = pd.read_csv(url)

    dropped_columns = ['PassengerId','Name','SibSp','Parch','Ticket','Fare','Cabin','Embarked']

    for column in dropped_columns:

        titanic = titanic.drop(column, axis=1)

    for column in ["Age", "Sex", "Pclass"]:
        titanic = titanic[titanic[column].notna()]

    sexo = {'male': 0, "female": 1}
    titanic['Sex'] = titanic["Sex"].map(sexo)

    #print(titanic.head(6))

    X = titanic.drop("Survived", axis=1)
    y = titanic["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    clf_dt = DecisionTreeClassifier(random_state=1)
    clf_dt.fit(X_train, y_train)
    acc_dt =  clf_dt.score(X_test, y_test)
    y_pred = clf_dt.predict(X_test)
    #print(y_pred)
    #print(y_test)
    #print(acc_dt)

    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig('static/confusion_matrix.png')  # Salva a matriz de confusão como uma imagem
    plt.close()  # Fecha o plot para evitar que seja exibido diretamente


    acuracia = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='macro')
    precisao = precision_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')


    print(acuracia)
    print(precisao)
    print(recall)
    print(f1)


    

    results = {
        "acuracia": acuracia,
        "precisao": precisao,
        "recall": recall,
        "f1score": f1
    }

    return results