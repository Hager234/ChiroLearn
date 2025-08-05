import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import matthews_corrcoef, f1_score
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier
from imblearn.combine import SMOTETomek
from skopt import BayesSearchCV


def load_data():
    data = pd.read_excel('metadata.xls', header=None)
    X = data.iloc[1:, 1:15]
    y = data.iloc[1:, 17]
    return X, y


def resample_smt(X, y):
    smt = SMOTETomek(random_state=42)
    X_res, y_res = smt.fit_resample(X, y)
    return (X_res, y_res)


def create_pipeline(classifier):
    if classifier == 'BaggingClassifier':
        steps = [
            ('scaler', StandardScaler()),
            ('classifier', BaggingClassifier(
                estimator=SVC(C=1.0, kernel='rbf'),
                n_estimators=100,
                random_state=42
            ))
        ]
    elif classifier == 'GradientBoostingClassifier':
        steps = [
            ('scaler', StandardScaler()),
            ('classifier', GradientBoostingClassifier(
                n_estimators=100,
                random_state=42
            ))
        ]
    return Pipeline(steps)


def perform_grid_search(X_train, y_train, pipeline, k, classifier):
    if classifier == 'BaggingClassifier':
        param_grid = {
            'classifier__n_estimators': [100]
        }
    elif classifier == 'GradientBoostingClassifier':
        param_grid = {
            'classifier__max_depth': [10],
            'classifier__max_features': ['sqrt'],
            'classifier__n_estimators': [100]
        }

    from sklearn.model_selection import GridSearchCV  # Instead of BayesSearchCV



    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=k,
        scoring=['accuracy', 'f1_macro', 'matthews_corrcoef'],
        n_jobs=-1,
        refit='accuracy',
        verbose=2
    )
    grid_search.fit(X_train, y_train)
    return grid_search


if __name__ == "__main__":

    X, y = load_data()

    label_enc = preprocessing.LabelEncoder()
    y = label_enc.fit_transform(y)

    x_train, y_train = resample_smt(X, y)

    list_of_ks = [5, 10, 20]
    list_of_classifiers = ["BaggingClassifier", "GradientBoostingClassifier"]

    for k in list_of_ks:
        summary_data = []
        for classifier in list_of_classifiers:
            pipe = create_pipeline(classifier)
            best_classifier = perform_grid_search(x_train, y_train, pipe, k, classifier)

            file_name = f'{classifier}_k{k}_exp.pkl'
            with open(file_name, 'wb') as file:
                pickle.dump(best_classifier, file)

            df = pd.DataFrame(best_classifier.cv_results_)

            excel_file_path = f'cv_results_k{k}_{classifier}_exp.xlsx'
            with pd.ExcelWriter(excel_file_path) as writer:
                df.to_excel(writer, index=False)

            best_index = best_classifier.best_index_

            mean_acc = df.loc[best_index, 'mean_test_accuracy']
            std_acc = df.loc[best_index, 'std_test_accuracy']
            mean_f1 = df.loc[best_index, 'mean_test_f1_macro']
            std_f1 = df.loc[best_index, 'std_test_f1_macro']
            mean_mcc = df.loc[best_index, 'mean_test_matthews_corrcoef']
            std_mcc = df.loc[best_index, 'std_test_matthews_corrcoef']

            summary_data.append({
                'Folds': k,
                'Classifier': classifier,
                'Best Parameters': best_classifier.best_params_,
                'Mean Accuracy': mean_acc,
                'Std Accuracy': std_acc,
                'Mean F1 (Macro)': mean_f1,
                'Std F1 (Macro)': std_f1,
                'Mean MCC': mean_mcc,
                'Std MCC': std_mcc,
            })

            print(
                f'k: {k}, Classifier: {classifier}, Accuracy: {mean_acc}, std of accuracy: {std_acc}, '
                f'f1: {mean_f1}, std of f1: {std_f1}, MCC: {mean_mcc}, std of mcc: {std_mcc}, '
                f'Best parameters: {best_classifier.best_params_}'
            )

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(f'summary_{k}_gridsearch.xlsx', index=False)
        print("\nSummary of Grid Search Results:")
        print(summary_df)
