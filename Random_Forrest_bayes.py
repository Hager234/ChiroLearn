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

    # Feature Importance (Method 1: Permutation)
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    perm_importances = []
    for train_idx, test_idx in kf.split(x_train, y_train):
        model = clone(best_classifier.best_estimator_)
        model.fit(x_train.iloc[train_idx], y_train[train_idx])
        result = permutation_importance(model, x_train.iloc[test_idx], y_train[test_idx],
                                        n_repeats=10, scoring='accuracy', random_state=42)
        perm_importances.append(result.importances_mean)

    perm_importance_cv = np.mean(perm_importances, axis=0)

    # Method 2: One Feature Accuracy CV
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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(f"feature_importances_Random  _{timestamp}.xlsx")
    importance_df.to_excel(out_path, index=False)
    print(f"✔ Raw feature importances written to {out_path.resolve()}")

    # Normalized Plot
    importance_df_normalized = importance_df.copy()
    for col in ['Accuracy_CV (Testing)', '1F_CV (Testing)', 'MCC_CV (Testing)']:
        importance_df_normalized[col] /= importance_df_normalized[col].max()

    plt.figure(figsize=(14, 7))
    for method in ['Accuracy_CV (Testing)', '1F_CV (Testing)', 'MCC_CV (Testing)']:
        plt.plot(importance_df_normalized['Feature'], importance_df_normalized[method], marker='o', label=method)
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Features')
    plt.ylabel('Normalized Importance')
    plt.title('Normalized Feature Importances')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Raw Importance Plot
    plt.figure(figsize=(14, 7))
    for method in ['Accuracy_CV (Testing)', '1F_CV (Testing)', 'MCC_CV (Testing)']:
        plt.plot(importance_df['Feature'], importance_df[method], marker='o', label=method)
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Features')
    plt.ylabel('Raw Importance')
    plt.title('Raw Feature Importances')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
