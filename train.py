"""Reproduce the concrete compressive strength machine-learning mini-project.

Requirements: pip install -r requirements.txt
Run: python train.py
Downloads the public UCI dataset (internet required on first run).
"""
from pathlib import Path
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_validate, train_test_split

ROOT = Path(__file__).resolve().parent
SEED = 42

def metrics(y_true, y_pred):
    return {
        "MAE_MPa": float(mean_absolute_error(y_true, y_pred)),
        "RMSE_MPa": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }

def make_models():
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, random_state=SEED, n_jobs=-1
        ),
    }

def main():
    # Original data: 1,030 observations and 8 explanatory variables.
    dataset = fetch_ucirepo(id=165)
    X_original = dataset.data.features.copy()
    target = dataset.data.targets.iloc[:, 0].to_numpy()
    df = X_original.copy()
    df["Strength"] = target
    duplicate_count = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    X = df.drop(columns=["Strength"])
    y = df["Strength"]

    print(f"Original samples: {len(X_original)}")
    print(f"Exact duplicate rows removed: {duplicate_count}")
    print(f"Samples after cleaning: {len(df)}")
    if len(df) != 1005:
        print("NOTE: dataset count differs from the supplied classroom run; check UCI version.")

    # 80/20 random holdout: same settings as the course notebook.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )
    holdout = {}
    fitted = {}
    for name, model in make_models().items():
        model.fit(X_train, y_train)
        fitted[name] = model
        holdout[name] = metrics(y_test, model.predict(X_test))
    print("\n=== Random 80/20 holdout ===")
    print(pd.DataFrame(holdout).T.round(4).to_string())

    # Save actual vs predicted chart for the random holdout.
    preds = {
        "Linear Regression": fitted["Linear Regression"].predict(X_test),
        "Random Forest": fitted["Random Forest"].predict(X_test),
    }
    bounds = [min(y_test.min(), *(p.min() for p in preds.values())),
              max(y_test.max(), *(p.max() for p in preds.values()))]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (name, p), color in zip(
        axes, preds.items(), ["steelblue", "darkorange"]
    ):
        ax.scatter(y_test, p, alpha=0.6, s=25, color=color)
        ax.plot(bounds, bounds, "k--", linewidth=1.5, label="Perfect prediction")
        ax.set(xlim=bounds, ylim=bounds, xlabel="Actual Strength (MPa)",
               ylabel="Predicted Strength (MPa)", title=name)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT / "Actual_vs_Predicted.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # A single 80/20 group split: whole mixture compositions stay together.
    mixture_cols = [column for column in X.columns if column != "Age"]
    groups = X.groupby(mixture_cols, sort=False).ngroup()
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    X_g_train, X_g_test = X.iloc[train_idx], X.iloc[test_idx]
    y_g_train, y_g_test = y.iloc[train_idx], y.iloc[test_idx]
    assert not set(groups.iloc[train_idx]).intersection(set(groups.iloc[test_idx]))

    group_models = make_models()
    group_holdout = {}
    for name, model in group_models.items():
        model.fit(X_g_train, y_g_train)
        group_holdout[name] = metrics(y_g_test, model.predict(X_g_test))
    print(f"\nUnique mixtures: {groups.nunique()}")
    print(f"Group split samples: train={len(train_idx)}, test={len(test_idx)}")
    print("=== Group-based 80/20 holdout ===")
    print(pd.DataFrame(group_holdout).T.round(4).to_string())

    # GroupKFold: each mixture appears in the validation set for one fold only.
    cv = GroupKFold(n_splits=5)
    scoring = {
        "MAE": "neg_mean_absolute_error",
        "RMSE": "neg_root_mean_squared_error",
        "R2": "r2",
    }
    cv_summary = {}
    for name, model in make_models().items():
        scores = cross_validate(
            model, X, y, groups=groups, cv=cv, scoring=scoring, n_jobs=1
        )
        cv_summary[name] = {
            "MAE_mean_MPa": float(-scores["test_MAE"].mean()),
            "MAE_std_MPa": float(scores["test_MAE"].std()),
            "RMSE_mean_MPa": float(-scores["test_RMSE"].mean()),
            "R2_mean": float(scores["test_R2"].mean()),
            "R2_std": float(scores["test_R2"].std()),
        }
    print("\n=== GroupKFold (5 folds) ===")
    print(pd.DataFrame(cv_summary).T.round(4).to_string())

    # Permutation importance on the group-held-out mixtures (not causal effects).
    perm = permutation_importance(
        group_models["Random Forest"], X_g_test, y_g_test,
        n_repeats=10, random_state=SEED,
        scoring="neg_mean_absolute_error", n_jobs=-1,
    )
    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance_MAE_MPa": perm.importances_mean,
        "Std_MPa": perm.importances_std,
    }).sort_values("Importance_MAE_MPa", ascending=False)
    importance.to_csv(ROOT / "permutation_importance.csv", index=False)
    plot_df = importance.sort_values("Importance_MAE_MPa")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(plot_df["Feature"], plot_df["Importance_MAE_MPa"],
            xerr=plot_df["Std_MPa"], color="steelblue", alpha=0.8, capsize=3)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(xlabel="Increase in MAE after permutation (MPa)",
           ylabel="Input Features", title="Random Forest - Permutation Feature Importance")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(ROOT / "Feature_Importance.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Refit final inference model on all cleaned observations.
    final_model = make_models()["Random Forest"]
    final_model.fit(X, y)
    joblib.dump({"model": final_model, "features": X.columns.tolist()},
                ROOT / "concrete_rf_model.joblib")

    report = {
        "original_n": len(X_original),
        "duplicate_rows_removed": duplicate_count,
        "cleaned_n": len(df),
        "unique_mixtures": int(groups.nunique()),
        "random_holdout": holdout,
        "group_holdout": group_holdout,
        "five_fold_group_cv": cv_summary,
        "seed": SEED,
        "n_estimators": 200,
    }
    (ROOT / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nSaved: concrete_rf_model.joblib, metrics.json, ")
    print("       Actual_vs_Predicted.png, Feature_Importance.png,")
    print("       permutation_importance.csv")

if __name__ == "__main__":
    main()
