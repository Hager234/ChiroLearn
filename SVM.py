import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer, accuracy_score, f1_score, matthews_corrcoef
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from imblearn.combine import SMOTETomek
import pickle
from datetime import datetime
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from pathlib import Path






def load_data():
    data = pd.read_excel('metadata.xls', header=None)
    X = data.iloc[1:, 1:15]
    y = data.iloc[1:, 17]
    return X, y


def resample_smt(X, y):
    smt = SMOTETomek(random_state=42)
    X_res, y_res = smt.fit_resample(X, y)
    return X_res, y_res


def create_svm_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', SVC(kernel='rbf', C=10, gamma=1))
    ])


def pearson_with_ci(x, y):
    r, p = pearsonr(x, y)
    stderr = 1.0 / np.sqrt(len(x) - 3)
    delta = 1.96 * stderr
    z = np.arctanh(r)
    lo = np.tanh(z - delta)
    hi = np.tanh(z + delta)
    return r, p, lo, hi


def generate_pearson_correlation_table(X, y, model, feature_labels):
    results = []

    y_pred = model.predict(X)
    r_model, p_model, lo_model, hi_model = pearson_with_ci(y_pred, y)
    results.append(['SVM Prediction', r_model, p_model, lo_model, hi_model])

    for col, label in zip(X.columns, feature_labels):
        r, p, lo, hi = pearson_with_ci(X[col], y)
        results.append([label, r, p, lo, hi])

    df = pd.DataFrame(results, columns=['Feature', 'Pearson r', 'p-value', '95% CI Lower', '95% CI Upper'])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(f"pearson_correlation_{timestamp}.xlsx")
    df.to_excel(path, index=False)
    print(f"Pearson correlation results saved to {path.resolve()}")
    return df




def plot_pearson_for_features_and_model(X, y, model, feature_labels):
    # Compute Pearson r for each feature
    feature_rs = [pearsonr(X[col], y)[0] for col in X.columns]
    r_model = pearsonr(model.predict(X), y)[0]

    # Combine values and labels
    all_rs = feature_rs + [r_model]
    all_labels = list(feature_labels) + ['SVM Prediction']
    all_colors = ['#96c7dc'] * len(feature_rs) + ['#fcb45c']  # blue for features, orange for model

    # Sort by Pearson r
    sorted_data = sorted(zip(all_rs, all_labels, all_colors), key=lambda x: x[0])
    sorted_rs, sorted_labels, sorted_colors = zip(*sorted_data)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(sorted_rs)), sorted_rs, color=sorted_colors, edgecolor='black', linewidth=1.5)

    # Set x-ticks
    ax.set_xticks(range(len(sorted_rs)))
    ax.set_xticklabels(sorted_labels, rotation=45, ha='right')

    ax.set_ylabel('Pearson r', fontsize=12)
    ax.set_xlabel('Features', fontsize=12)
    ax.set_title('Pearson Correlation of Features and SVM Predictions', fontsize=14, weight='bold')
    ax.set_ylim(-0.20, 1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.tick_params(axis='both', which='major', labelsize=10)

    # Annotate bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4 if height >= 0 else -12),
                    textcoords="offset points",
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=10, color='black')

    plt.tight_layout()
    plt.show()

custom_labels = [
        'T0_s_', 'T1_s_', 'T1_N_', 'T2_s_', 'T2_N_', 'T3_s_', 'T3_N_', 'T4_s_', 'T4_N_',
        'ThrustDuration_s_', 'Avg_ThrustSpeed_N_s_', 'Max_ThrustSpeed_N_s_',
        'PreloadDosage_N_s_', 'ThrustDosage_N_s_'
    ]


if __name__ == "__main__":

    X, y = load_data()
    y = LabelEncoder().fit_transform(y)
    x_train, y_train = resample_smt(X, y)

    list_of_ks = [5, 10, 20]
    summary_data = []

    for k in list_of_ks:
        pipe = create_svm_pipeline()

        param_grid = {
            'classifier__C': [10],
            'classifier__gamma': [1],
            'classifier__kernel': ['rbf']
        }

        grid_search = GridSearchCV(pipe, param_grid, cv=k,
                                   scoring=['accuracy', 'f1_macro', 'matthews_corrcoef'],
                                   refit='accuracy', verbose=2, n_jobs=-1)
        grid_search.fit(x_train, y_train)

        best_classifier = grid_search.best_estimator_

        with open(f'svm_classifier_k{k}.pkl', 'wb') as file:
            pickle.dump(best_classifier, file)

        # Save cv results from the grid search (not best_classifier):
        pd.DataFrame(grid_search.cv_results_).to_excel(f'cv_results_k{k}_svm_exp.xlsx', index=False)

        # Append summary from grid_search.cv_results_:
        summary_data.append({
            'Folds': k,
            'Best Parameters': grid_search.best_params_,
            'Mean Accuracy': grid_search.cv_results_['mean_test_accuracy'][grid_search.best_index_],
            'Std Accuracy': grid_search.cv_results_['std_test_accuracy'][grid_search.best_index_],
            'Mean F1 (Macro)': grid_search.cv_results_['mean_test_f1_macro'][grid_search.best_index_],
            'Std F1 (Macro)': grid_search.cv_results_['std_test_f1_macro'][grid_search.best_index_],
            'Mean MCC': grid_search.cv_results_['mean_test_matthews_corrcoef'][grid_search.best_index_],
            'Std MCC': grid_search.cv_results_['std_test_matthews_corrcoef'][grid_search.best_index_],
        })


        corr_df = generate_pearson_correlation_table(x_train, y_train, best_classifier, custom_labels)
        plot_pearson_for_features_and_model(x_train, y_train, best_classifier, custom_labels)

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_excel('svm_summary_fixedparams_gridsearch.xlsx', index=False)
    print("\nSummary of cross-validation results:")
    print(summary_df)
    



    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    n_features = x_train.shape[1]

    perm_acc_importances = []
    perm_mcc_importances = []
    perm_f1_importances = []

    # Method 1: Permutation Importance for Accuracy, MCC, F1
    for train_idx, test_idx in kf.split(x_train, y_train):
        model = clone(best_classifier)
        model.fit(x_train.iloc[train_idx], y_train[train_idx])

        acc_result = permutation_importance(model, x_train.iloc[test_idx], y_train[test_idx],
                                            scoring='accuracy', n_repeats=10, random_state=42)
        mcc_result = permutation_importance(model, x_train.iloc[test_idx], y_train[test_idx],
                                            scoring='matthews_corrcoef', n_repeats=10, random_state=42)
        f1_result = permutation_importance(model, x_train.iloc[test_idx], y_train[test_idx],
                                           scoring='f1_macro', n_repeats=10, random_state=42)

        perm_acc_importances.append(acc_result.importances_mean)
        perm_mcc_importances.append(mcc_result.importances_mean)
        perm_f1_importances.append(f1_result.importances_mean)

    perm_acc = np.mean(perm_acc_importances, axis=0)
    perm_mcc = np.mean(perm_mcc_importances, axis=0)
    perm_f1 = np.mean(perm_f1_importances, axis=0)

    # One Feature CV (Accuracy)
    one_feature_acc = []
    for i in range(n_features):
        clf = SVC(kernel='rbf', C=10, gamma=1)
        scores = cross_val_score(clf, x_train.iloc[:, i].values.reshape(-1, 1), y_train, cv=5, scoring='accuracy')
        one_feature_acc.append(np.mean(scores))

    # One Feature CV (MCC & F1)
    one_feature_mcc = []
    one_feature_f1 = []
    for i in range(n_features):
        clf = SVC(kernel='rbf', C=10, gamma=1)
        # One Feature CV (MCC)
        mcc = cross_val_score(clf, x_train.iloc[:, i].values.reshape(-1, 1), y_train, cv=5,
                              scoring=make_scorer(matthews_corrcoef))
        # One Feature CV (F1)
        f1 = cross_val_score(clf, x_train.iloc[:, i].values.reshape(-1, 1), y_train, cv=5, scoring='f1_macro')
        one_feature_mcc.append(np.mean(mcc))
        one_feature_f1.append(np.mean(f1))

    # Create DataFrame

    importance_df = pd.DataFrame({
        'Feature': custom_labels,
        'Perm_ACC': perm_acc,
        'Perm_MCC': perm_mcc,
        'Perm_F1': perm_f1,
        '1F_ACC': one_feature_acc,
        '1F_MCC': one_feature_mcc,
        '1F_F1': one_feature_f1
    })
    print(importance_df)

