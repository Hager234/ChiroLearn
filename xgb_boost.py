from xgboost import XGBClassifier, Booster
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import matthews_corrcoef, f1_score
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f_oneway, ttest_ind
import seaborn as sns
import pickle
from sklearn.metrics import make_scorer, matthews_corrcoef
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR

from sklearn.metrics import r2_score
from sklearn.preprocessing import FunctionTransformer, StandardScaler
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind


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

    # Method 1: Permutation Importance
    from sklearn.base import clone
    from sklearn.model_selection import StratifiedKFold
    import numpy as np

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    perm_importances = []

    for train_idx, test_idx in kf.split(x_train, y_train):
        # Clone the best estimator (fresh model)
        model = clone(best_classifier.best_estimator_)

        # Fit on training fold
        model.fit(x_train.iloc[train_idx], y_train[train_idx])

        # Compute permutation importance on the test fold
        result = permutation_importance(
            model,
            x_train.iloc[test_idx],
            y_train[test_idx],
            n_repeats=10,
            scoring='accuracy',
            random_state=42
        )
        perm_importances.append(result.importances_mean)

    # Average importance over all folds
    perm_importance_cv = np.mean(perm_importances, axis=0)

    # Print result
    print("Permutation Importance (CV-based on held-out test folds):")
    print(perm_importance_cv)

    # Method 2: 1F (One Feature Cross-Validation)
    one_feature_scores = []
    for i in range(x_train.shape[1]):
        clf = XGBClassifier(objective='multi:softmax', use_label_encoder=False, eval_metric='mlogloss')
        score = cross_val_score(clf, x_train.iloc[:, i].values.reshape(-1, 1), y_train, cv=5, scoring='accuracy')
        one_feature_scores.append(np.mean(score))

    # Method 3: MCC
    mcc_importances = []
    for i in range(x_train.shape[1]):
        clf = XGBClassifier(objective='multi:softmax', use_label_encoder=False, eval_metric='mlogloss')
        scores = cross_val_score(clf, x_train.iloc[:, i].values.reshape(-1, 1), y_train, cv=5,
                                 scoring=make_scorer(matthews_corrcoef))
        mcc_importances.append(np.mean(scores))

    custom_labels = [
        'T0_s_', 'T1_s_', 'T1_N_', 'T2_s_', 'T2_N_', 'T3_s_', 'T3_N_', 'T4_s_', 'T4_N_',
        'ThrustDuration_s_', 'Avg_ThrustSpeed_N_s_', 'Max_ThrustSpeed_N_s_', 'PreloadDosage_N_s_',
        'ThrustDosage_N_s_'
    ]

    importance_df = pd.DataFrame({
        'Feature': custom_labels,
        'Accuracy_CV (Testing)': perm_importance_cv,
        '1F_CV (Testing)': one_feature_scores,
        'MCC_CV (Testing)': mcc_importances
    })

    importance_df['Avg_Importance'] = importance_df[
        ['Accuracy_CV (Testing)', '1F_CV (Testing)', 'MCC_CV (Testing)']
    ].mean(axis=1)

    importance_df = importance_df.sort_values(by='Avg_Importance', ascending=False).reset_index(drop=True)
    print(importance_df)
    from datetime import datetime

    # Add timestamp to avoid overwrite conflicts
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(f"feature_importances_{timestamp}.xlsx")

    importance_df.to_excel(out_path, index=False, engine="openpyxl")

    print(f"✔ Raw feature importances written to {out_path.resolve()}")

    # Normalize for plotting
    importance_df_normalized = importance_df.copy()
    for col in ['Accuracy_CV (Testing)', '1F_CV (Testing)', 'MCC_CV (Testing)']:
        importance_df_normalized[col] = importance_df_normalized[col] / importance_df_normalized[col].max()

    # 🔥 Normalized Plotting
    plt.figure(figsize=(14, 7))
    for method in ['Accuracy_CV (Testing)', '1F_CV (Testing)', 'MCC_CV (Testing)']:
        plt.plot(
            importance_df_normalized['Feature'],
            importance_df_normalized[method],
            marker='o',
            label=method
        )

    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Features')
    plt.ylabel('Normalized Importance')
    plt.title('Normalized Feature Importances by Different Methods (Training vs Testing)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 🔥 Raw Importance Plotting
    plt.figure(figsize=(14, 7))
    for method in ['Accuracy_CV (Testing)', '1F_CV (Testing)', 'MCC_CV (Testing)']:
        plt.plot(
            importance_df['Feature'],
            importance_df[method],
            marker='o',
            label=method
        )

    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Features')
    plt.ylabel('Raw Importance')
    plt.title('Raw Feature Importances by Different Methods (Testing)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


