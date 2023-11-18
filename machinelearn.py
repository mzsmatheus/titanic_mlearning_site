#imports de ciencia de dados e machine learning
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

#imports dos classificadores
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier

matplotlib.use('agg')
def treinamentoteste(classifier_type, param1, param2, param3):

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
    print(classifier_type)
    print(param1)
    print(param2)
    print(param3)

    X = titanic.drop("Survived", axis=1)
    y = titanic["Survived"]

    size = X.shape
    size_str = f'Dimensões do DataSet: {size[0]} linhas, {size[1]} colunas'

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    if classifier_type == 'DT':
        clf_dt = DecisionTreeClassifier(min_samples_leaf=param1, max_depth=param2, min_samples_split=param3)
        clf_dt.fit(X_train, y_train)
        acc_dt = clf_dt.score(X_test, y_test)
        y_pred = clf_dt.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.savefig('static/confusion_matrix.png')
        plt.close()

        acuracia = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred, average='macro')
        precisao = precision_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')

        results = {
            "tamanho:": size_str,
            "classif": "Decision Tree",
            "media": "macro",
            "paramt1": f"min_samples_leaf: {param1}",
            "paramt2": f"max_depth: {param2}",
            "paramt3": f"min_samples_split: {param3}",
            "acuracia": acuracia,
            "precisao": precisao,
            "recall": recall,
            "f1score": f1
        }

        return results

    elif classifier_type == 'RF': 
        clf_rfc = RandomForestClassifier(max_depth=param1, n_estimators=param2, min_samples_split=param3)
        clf_rfc.fit(X_train, y_train)
        acc_rfc = clf_rfc.score(X_test, y_test)
        y_pred = clf_rfc.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.savefig('static/confusion_matrix.png')
        plt.close()

        acuracia = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred, average='macro')
        precisao = precision_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')

        results = {
            "tamanho:": size_str,
            "classif": "Random Forest",
            "media": "macro",
            "paramt1": f"max_depth: {param1}",
            "paramt2": f"n_estimators: {param2}",
            "paramt3": f"min_samples_split: {param3}",
            "acuracia": acuracia,
            "precisao": precisao,
            "recall": recall,
            "f1score": f1
        }

        return results

    elif classifier_type == 'SVC':
        clf_svc = make_pipeline(StandardScaler(), SVC(C=param1, gamma=param3, degree=param2))
        clf_svc.fit(X_train, y_train)
        acc_svc = clf_svc.score(X_test, y_test)
        y_pred = clf_svc.predict(X_test)
        

        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.savefig('static/confusion_matrix.png')
        plt.close()

        acuracia = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred, average='macro')
        precisao = precision_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')

        results = {
            "tamanho:": size_str,
            "classif": "Support Vector Classifier",
            "media": "macro",
            "paramt1": f"C: {param1}",
            "paramt2": f"degree: {param2}",
            "paramt3": f"gamma: {param3}",
            "acuracia": acuracia,
            "precisao": precisao,
            "recall": recall,
            "f1score": f1
        }

        return results

    elif classifier_type == 'KNN':
        clf_knn = KNeighborsClassifier(n_neighbors=param1, leaf_size=param3, n_jobs=param2)
        clf_knn.fit(X_train, y_train)
        acc_knn = clf_knn.score(X_test, y_test)
        y_pred = clf_knn.predict(X_test)
        

        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.savefig('static/confusion_matrix.png')
        plt.close()

        acuracia = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred, average='macro')
        precisao = precision_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')

        results = {
            "tamanho:": size_str,
            "classif": "K-Nearest Neighbors",
            "media": "macro",
            "paramt1": f"n_neighbors: {param1}",
            "paramt2": f"n_jobs: {param2}",
            "paramt3": f"leaf_size: {param3}",
            "acuracia": acuracia,
            "precisao": precisao,
            "recall": recall,
            "f1score": f1
        }

        return results

    elif classifier_type == 'GBM': 
        clf_gbm = GradientBoostingClassifier(learning_rate=param1, max_depth=param2, n_estimators=param3) 
        clf_gbm.fit(X_train, y_train)
        acc_gbm = clf_gbm.score(X_test, y_test)
        y_pred = clf_gbm.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.savefig('static/confusion_matrix.png')
        plt.close()

        acuracia = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred, average='macro')
        precisao = precision_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')

        results = {
            "tamanho:": size_str,
            "classif": "Gradient Boosting Classifier",
            "media": "macro",
            "paramt1": f"learning_rate: {param1}",
            "paramt2": f"max_depth: {param2}",
            "paramt3": f"n_estimators: {param3}",
            "acuracia": acuracia,
            "precisao": precisao,
            "recall": recall,
            "f1score": f1
        }

        return results

    else:
        return {}