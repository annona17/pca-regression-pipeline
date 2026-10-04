import numpy as np
import pandas as pd

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, KFold, cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

diabetes = load_diabetes(as_frame=True)
df = diabetes.frame.copy()
target_col = "target"
X = df.drop(columns=[target_col])
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("scaler", StandardScaler()),
])
preprocessor = ColumnTransformer(transformers=[("num", numeric_transformer, numeric_features)])
n_expanded = preprocessor.fit_transform(X_train).shape[1]

cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scoring = {
    "MAE": make_scorer(mean_absolute_error, greater_is_better=False),
    "RMSE": make_scorer(lambda yt, yp: np.sqrt(mean_squared_error(yt, yp)), greater_is_better=False),
    "R2": make_scorer(r2_score),
}

def evaluate_model(name, pipeline, X=X_train, y=y_train):
    results = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    return {
        "Model": name,
        "MAE_mean": -results["test_MAE"].mean(), "MAE_std": results["test_MAE"].std(),
        "RMSE_mean": -results["test_RMSE"].mean(), "RMSE_std": results["test_RMSE"].std(),
        "R2_mean": results["test_R2"].mean(), "R2_std": results["test_R2"].std(),
    }

baseline_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", DummyRegressor(strategy="mean"))])
baseline_result = evaluate_model("Baseline (mean)", baseline_pipeline)

models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0, random_state=RANDOM_STATE),
    "Lasso": Lasso(alpha=0.01, max_iter=20000, random_state=RANDOM_STATE),
    "KNN Regressor": KNeighborsRegressor(n_neighbors=10),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    "SVR (RBF)": SVR(kernel="rbf", C=10, epsilon=0.1),
}

results_no_pca = [baseline_result]
for name, model in models.items():
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    results_no_pca.append(evaluate_model(name, pipe))
df_no_pca = pd.DataFrame(results_no_pca)

X_train_pre = preprocessor.fit_transform(X_train)
n_features_after_prep = X_train_pre.shape[1]
pca_full = PCA(n_components=n_features_after_prep, random_state=RANDOM_STATE).fit(X_train_pre)
cum_var = np.cumsum(pca_full.explained_variance_ratio_)
n_components_95 = int(np.argmax(cum_var >= 0.95) + 1)

results_pca = []
for name, model in models.items():
    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("pca", PCA(n_components=n_components_95, random_state=RANDOM_STATE)),
        ("model", model),
    ])
    results_pca.append(evaluate_model(name + " + PCA", pipe))
df_pca = pd.DataFrame(results_pca)

all_results = pd.concat([df_no_pca, df_pca], ignore_index=True)
all_results_sorted = all_results.sort_values("RMSE_mean").reset_index(drop=True)
best_row = all_results_sorted.iloc[0]

param_grids = {
    "Linear Regression": {"model__fit_intercept": [True, False]},
    "Ridge": {"model__alpha": [0.1, 1.0, 10.0, 50.0, 100.0]},
    "Lasso": {"model__alpha": [0.001, 0.01, 0.1, 1.0]},
    "KNN Regressor": {"model__n_neighbors": [5, 10, 15, 20], "model__weights": ["uniform", "distance"]},
    "Random Forest": {"model__n_estimators": [200, 400], "model__max_depth": [None, 12, 20], "model__min_samples_leaf": [1, 2, 4]},
    "Gradient Boosting": {"model__n_estimators": [100, 200], "model__max_depth": [2, 3, 4], "model__learning_rate": [0.05, 0.1]},
    "SVR (RBF)": {"model__C": [1, 10, 50], "model__epsilon": [0.05, 0.1, 0.2]},
}
best_model_label = best_row["Model"]
use_pca = best_model_label.endswith(" + PCA")
base_model_name = best_model_label.replace(" + PCA", "")
steps = [("preprocessor", preprocessor)]
if use_pca:
    steps.append(("pca", PCA(n_components=n_components_95, random_state=RANDOM_STATE)))
steps.append(("model", models[base_model_name]))
best_pipeline = Pipeline(steps=steps)
param_grid = param_grids[base_model_name]

grid_search = GridSearchCV(best_pipeline, param_grid=param_grid, scoring="neg_root_mean_squared_error", cv=cv, n_jobs=-1)
grid_search.fit(X_train, y_train)

final_model = grid_search.best_estimator_
final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)
final_mae = mean_absolute_error(y_test, y_pred)
final_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
final_r2 = r2_score(y_test, y_pred)

baseline_pipeline.fit(X_train, y_train)
y_pred_base = baseline_pipeline.predict(X_test)
base_mae = mean_absolute_error(y_test, y_pred_base)
base_rmse = np.sqrt(mean_squared_error(y_test, y_pred_base))
base_r2 = r2_score(y_test, y_pred_base)

print("=== n_expanded (poly features) ===", n_expanded)
print("=== n_components_95 ===", n_components_95)
print("\n=== ALL RESULTS SORTED (full precision) ===")
for _, r in all_results_sorted.iterrows():
    print(f"{r['Model']:<28} MAE={r['MAE_mean']:.4f}±{r['MAE_std']:.4f}  RMSE={r['RMSE_mean']:.4f}±{r['RMSE_std']:.4f}  R2={r['R2_mean']:.4f}±{r['R2_std']:.4f}")

print("\n=== BEST MODEL ===", best_model_label)
print("GridSearch best params:", grid_search.best_params_)
print("GridSearch best CV RMSE:", -grid_search.best_score_)

print("\n=== FINAL TEST SET ===")
print(f"Baseline:  MAE={base_mae:.4f} RMSE={base_rmse:.4f} R2={base_r2:.4f}")
print(f"Tuned({best_model_label}): MAE={final_mae:.4f} RMSE={final_rmse:.4f} R2={final_r2:.4f}")

all_results_sorted.to_csv("report_assets/all_results.csv", index=False)
df_no_pca.to_csv("report_assets/no_pca.csv", index=False)
df_pca.to_csv("report_assets/pca.csv", index=False)

# Bang so sanh truc tiep co / khong PCA cho tung mo hinh
impact = df_no_pca[df_no_pca["Model"] != "Baseline (mean)"].merge(
    df_pca.assign(base=df_pca["Model"].str.replace(" + PCA", "", regex=False)),
    left_on="Model", right_on="base", suffixes=("_noPCA", "_PCA"),
)
impact["RMSE_change_%"] = (impact["RMSE_mean_PCA"] - impact["RMSE_mean_noPCA"]) / impact["RMSE_mean_noPCA"] * 100
impact.to_csv("report_assets/pca_impact.csv", index=False)

import json
summary = {
    "n_samples": int(df.shape[0]),
    "n_features_original": int(X.shape[1]),
    "n_features_expanded": int(n_expanded),
    "n_components_95": int(n_components_95),
    "best_model_label": best_model_label,
    "best_params": grid_search.best_params_,
    "baseline_test": {"MAE": base_mae, "RMSE": base_rmse, "R2": base_r2},
    "final_test": {"MAE": final_mae, "RMSE": final_rmse, "R2": final_r2},
}
with open("report_assets/summary.json", "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print("\nSaved CSVs + summary.json to report_assets/")
