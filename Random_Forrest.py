import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import matthews_corrcoef, make_scorer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from imblearn.combine import SMOTETomek
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from pathlib import Path
from datetime import datetime
import pickle


def load_data():
    data = pd.read_excel('metadata.xls', header=None)
    X = data.iloc[1:, 1:15]
    y = data.iloc[1:, 17]
    return X, y


def resample_smt(X, y):
    smt = SMOTETomek(random_state=42)
    X_res, y_res = smt.fit_resample(X, y)
    return X_res, y_res


def create_pipeline():
    steps = [
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier())
    ]
    return Pipeline(steps)


def perform_grid_search(X_train, y_train, pipeline, k):
    param_grid = {
        'classifier__max_depth': [None],
        'classifier__min_samples_split': [2, 4, 6, 10, 100],
        'classifier__max_leaf_nodes': [1000, None],
        'classifier__min_samples_leaf': [1, 10],
        'classifier__n_estimators': [50, 100, 200, 400],
        'classifier__max_features': [1, 4],
        'classifier__max_samples': [0.1, 0.5, 1]
    }

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=k,
        scoring={
            'accuracy': 'accuracy',
            'f1_macro': 'f1_macro',
            'matthews_corrcoef': make_scorer(matthews_corrcoef)
        },
        refit='accuracy',
        n_jobs=-1,
        verbose=2
    )

    grid_search.fit(X_train, y_train)
    return grid_search


if __name__ == "__main__":
    X, y = load_data()

    label_enc = LabelEncoder()
    y = label_enc.fit_transform(y)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=42)
    x_train, y_train = resample_smt(x_train, y_train)

    list_of_ks = [5, 10, 20]
    summary_data = []

    for k in list_of_ks:
        pipe = create_pipeline()
        best_classifier = perform_grid_search(x_train, y_train, pipe, k)

        file_name = f'xgb_k{k}_exp.pkl'
        with open(file_name, 'wb') as file:
            pickle.dump(best_classifier, file)

        df = pd.DataFrame(best_classifier.cv_results_)
        df.to_excel(f'cv_results_k{k}_rf_exp.xlsx', index=False)

        best_index = best_classifier.best_index_

        mean_acc = df.loc[best_index, 'mean_test_accuracy']
        std_acc = df.loc[best_index, 'std_test_accuracy']
        mean_f1 = df.loc[best_index, 'mean_test_f1_macro']
        std_f1 = df.loc[best_index, 'std_test_f1_macro']
        mean_mcc = df.loc[best_index, 'mean_test_matthews_corrcoef']
        std_mcc = df.loc[best_index, 'std_test_matthews_corrcoef']

        summary_data.append({
            'Folds': k,
            'Best Parameters': best_classifier.best_params_,
            'Mean Accuracy': mean_acc,
            'Std Accuracy': std_acc,
            'Mean F1 (Macro)': mean_f1,
            'Std F1 (Macro)': std_f1,
            'Mean MCC': mean_mcc,
            'Std MCC': std_mcc,
        })

        print(
            f'k: {k}, Classifier: RandomForest, Accuracy: {mean_acc}, std of accuracy: {std_acc}, '
            f'f1: {mean_f1}, std of f1: {std_f1}, MCC: {mean_mcc}, std of mcc: {std_mcc}, '
            f'Best parameters: {best_classifier.best_params_}'
        )

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_excel('random_summary_gridsearch.xlsx', index=False)
    print("\nSummary of Grid Search Results:")
    print(summary_df)

    
