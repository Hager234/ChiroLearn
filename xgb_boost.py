from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import matthews_corrcoef, f1_score
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import pickle


def load_data():

    data = pd.read_excel('metadata.xls', header=None)
    X = data.iloc[1:, 1:15]
    y = data.iloc[1:, 17]

    return X, y

# resampling
def resample_smt(X, y):
    smt = SMOTETomek(random_state=42)
    X_res, y_res = smt.fit_resample(X, y)
    return (X_res, y_res)


def create_pipeline():
    steps = [
        ('scaler', StandardScaler()),  # Scale the features
        ('classifier', XGBClassifier(objective='multi:softmax'))  # BaggingClassifier with SVM base estimator
    ]
    pipeline = Pipeline(steps)
    return pipeline


def perform_grid_search(X_train, y_train, pipeline, k):
    # Define hyperparameters to tune
    param_grid = {
        'classifier__n_estimators': [400],
        'classifier__max_depth': [6],
        'classifier__learning_rate': [0.20, 0.1],
        'classifier__min_child_weight': [1]
    }


    grid_search = GridSearchCV(pipeline, param_grid, cv=k, scoring=['accuracy', 'f1_macro', 'matthews_corrcoef'],
                               n_jobs=-1, refit='accuracy', verbose=2)

    grid_search.fit(X_train, y_train)

    grid_search.cv_results_

    return grid_search


if __name__ == "__main__":
    X, y = load_data()
    print(X)
    label_enc = preprocessing.LabelEncoder()
    y = label_enc.fit_transform(y)

    x_train, y_train = resample_smt(X, y)

    list_of_ks = [5, 10, 20]

    summary_data = []

    for k in list_of_ks:
        pipe = create_pipeline()
        best_classifier = perform_grid_search(x_train, y_train, pipe, k)

        print(best_classifier.best_estimator_.named_steps['classifier'].feature_importances_)

        file_name = f'xgb_classifier_k{k}.pkl'
        with open(file_name, 'wb') as file:
            pickle.dump(best_classifier, file)

        df = pd.DataFrame(best_classifier.cv_results_)
        df.to_excel(f'cv_results_k{k}_xgb_exp.xlsx', index=False)

        # Extract metrics
        summary_data.append({
            'Folds': k,
            'Best Parameters': best_classifier.best_params_,
            'Mean Accuracy': best_classifier.cv_results_['mean_test_accuracy'][best_classifier.best_index_],
            'Std Accuracy': best_classifier.cv_results_['std_test_accuracy'][best_classifier.best_index_],
            'Mean F1 (Macro)': best_classifier.cv_results_['mean_test_f1_macro'][best_classifier.best_index_],
            'Std F1 (Macro)': best_classifier.cv_results_['std_test_f1_macro'][best_classifier.best_index_],
            'Mean MCC': best_classifier.cv_results_['mean_test_matthews_corrcoef'][best_classifier.best_index_],
            'Std MCC': best_classifier.cv_results_['std_test_matthews_corrcoef'][best_classifier.best_index_],
        })

    summary_df = pd.DataFrame(summary_data)

    summary_df.to_excel('xgb_summary_gridsearch.xlsx', index=False)
    print("\nSummary of Grid Search Results:")
    print(summary_df)

   
