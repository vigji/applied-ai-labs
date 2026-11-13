"""Tests for the MLflow lab (Day 0). The notebook downloads this file and imports it as `tests`.

Each test takes what you wrote, checks it, and either prints that it passed or stops
with an AssertionError that says what is wrong. The tests use their own small made-up
wine table, so they do not depend on your train/test split.
"""
from __future__ import annotations

import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

COLUMNS = ["type", "fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides",
           "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"]
METRICS = {"accuracy", "balanced_accuracy", "precision", "recall", "roc_auc"}


def _wines(n=200, seed=0):
    """A made-up table with the columns of the real one, and a label that depends on alcohol."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, len(COLUMNS) - 1)), columns=COLUMNS[1:])
    X["alcohol"] = rng.normal(10.5, 1.2, n)
    X.insert(0, "type", rng.choice(["red", "white"], n))
    y = pd.Series((X["alcohol"] + rng.normal(0, 0.8, n) > 11.2).astype(int), name="good")
    return X, y


def _passed(name):
    print(f"All tests in `{name}` passed.")


# ---------------------------------------------------------------- Exercise 1: the pyfunc model

def test_features(WineModel):
    X, _ = _wines(20)
    before = X.copy()
    out = WineModel._features(X)
    assert out is not None, "_features returned None: it must return a DataFrame."
    assert isinstance(out, pd.DataFrame), f"_features must return a DataFrame, not {type(out).__name__}."
    assert X.equals(before), "_features changed its input. Work on a copy (df.copy())."
    assert list(out.columns) == list(X.columns), "_features must keep the same columns, in the same order."
    assert pd.api.types.is_numeric_dtype(out["type"]), "`type` is still text: turn red/white into numbers."
    assert set(out["type"].unique()) <= {0, 1}, "Encode `type` as 0 and 1."
    red = (X["type"] == "red").to_numpy()
    assert (out["type"].to_numpy()[red] == 1).all() and (out["type"].to_numpy()[~red] == 0).all(), \
        "Use red = 1, white = 0, so that everyone's models agree."
    assert out.drop(columns="type").equals(X.drop(columns="type")), "_features must leave the other columns as they are."
    _passed("test_features")


def test_fit(WineModel):
    X, y = _wines()
    model = WineModel(LogisticRegression(max_iter=500))
    try:
        returned = model.fit(X, y)
    except ValueError as e:
        raise AssertionError(f"fit failed on raw rows ({str(e)[:80]}...). Did you pass the rows through _features?") from None
    assert returned is model, "fit must return self, so that you can write WineModel(...).fit(X, y)."
    assert hasattr(model.estimator, "coef_"), "fit returned, but the estimator inside was not fitted."
    _passed("test_fit")


def test_predict(WineModel):
    X, y = _wines()
    model = WineModel(LogisticRegression(max_iter=500), threshold=0.5).fit(X, y)
    try:
        out = model.predict(None, X)
    except ValueError as e:
        raise AssertionError(f"predict failed on raw rows ({str(e)[:80]}...). Did you pass the rows through _features?") from None
    assert out is not None, "predict returned None: it must return a DataFrame."
    assert isinstance(out, pd.DataFrame), f"predict must return a DataFrame, not {type(out).__name__}."
    assert sorted(out.columns) == ["good", "p_good"], f"predict must return exactly the columns `good` and `p_good`, not {list(out.columns)}."
    assert len(out) == len(X), f"One output row per input row: got {len(out)} for {len(X)}."
    p = out["p_good"].to_numpy()
    assert ((p >= 0) & (p <= 1)).all(), "`p_good` must be a probability, between 0 and 1."
    expected = model.estimator.predict_proba(WineModel._features(X))[:, 1]
    assert np.allclose(p, expected), "`p_good` must be the estimator's probability of class 1 (predict_proba(...)[:, 1])."
    assert set(out["good"].unique()) <= {0, 1}, "`good` must be 0 or 1."
    assert (out["good"].to_numpy() == (p >= 0.5)).all(), "`good` must be 1 exactly when p_good >= threshold."
    model.threshold = 0.0
    assert (model.predict(None, X)["good"] == 1).all(), "With threshold 0 every wine is good: use self.threshold, not a fixed 0.5."
    model.threshold = 1.01
    assert (model.predict(None, X)["good"] == 0).all(), "With threshold above 1 no wine is good: use self.threshold."
    _passed("test_predict")


# ---------------------------------------------------------------- Exercises 2 and 3: logging runs

def _check_logged(run_id, features, params_expected=()):
    client = mlflow.MlflowClient()
    data = client.get_run(run_id).data
    assert data.params, "Nothing was logged as a parameter: use mlflow.log_params."
    for key in ("threshold", "features", *params_expected):
        assert key in data.params, f"Parameter `{key}` is missing. Logged: {sorted(data.params)}."
    for f in features:
        assert f in data.params["features"], f"The `features` parameter does not list `{f}`."
    missing = METRICS - set(data.metrics)
    assert not missing, f"Metrics missing: {sorted(missing)}. Use mlflow.log_metrics(m)."
    logged = [a.path for a in client.list_artifacts(run_id)]
    models = mlflow.search_logged_models(filter_string=f"source_run_id = '{run_id}'", output_format="list")
    assert models or "model" in logged, "No model was logged in this run: use mlflow.pyfunc.log_model(name='model', ...)."
    loaded = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
    sig = loaded.metadata.signature
    assert sig is not None, "The model has no signature: pass signature=infer_signature(inputs, outputs)."
    assert sig.inputs.input_names() == list(features), \
        f"The signature's inputs should be the training columns {list(features)}, not {sig.inputs.input_names()}."
    assert sig.outputs is not None and sorted(sig.outputs.input_names()) == ["good", "p_good"], \
        "The signature's outputs should be `good` and `p_good`: infer it from the output of predict."
    assert loaded.metadata.saved_input_example_info is not None, "No input example: pass input_example=a few training rows."
    return client, data, loaded


def test_first_run(run_id, X_train):
    _, data, loaded = _check_logged(run_id, list(X_train.columns), params_expected=("n_estimators",))
    out = loaded.predict(X_train.head(5))
    assert sorted(out.columns) == ["good", "p_good"], "The logged model does not return `good` and `p_good`."
    _passed("test_first_run")


def test_run_experiment(run_experiment):
    features = ["type", "alcohol"]
    result = run_experiment(LogisticRegression(max_iter=500), features=features, threshold=0.4, run_name="test-run_experiment")
    assert result is not None and len(result) == 2, "run_experiment must return (run_id, metrics)."
    run_id, m = result
    try:
        assert isinstance(m, dict) and METRICS <= set(m), "run_experiment must return the metrics dict as its second value."
        client, data, _ = _check_logged(run_id, features, params_expected=("estimator", "max_iter"))
        assert data.params["estimator"] == "LogisticRegression", \
            f"The `estimator` parameter should be the class name, LogisticRegression, not {data.params['estimator']!r}."
        assert float(data.params["threshold"]) == 0.4, "Log the threshold you were given, not a fixed one."
        assert "density" not in data.params["features"], "Log only the features you trained on."
    finally:
        mlflow.MlflowClient().delete_run(run_id)   # keep the test run out of your table
    _passed("test_run_experiment")


# ---------------------------------------------------------------- Exercise 4: the registry

def test_champion(name, expected_run_id):
    client = mlflow.MlflowClient()
    try:
        mv = client.get_model_version_by_alias(name, "champion")
    except mlflow.exceptions.MlflowException:
        raise AssertionError(f"`{name}` has no version with the alias `champion`.") from None
    assert mv.run_id == expected_run_id, \
        f"`champion` points to version {mv.version}, which comes from run {mv.run_id[:8]}, not from {expected_run_id[:8]}."
    loaded = mlflow.pyfunc.load_model(f"models:/{name}@champion")
    assert loaded.metadata.run_id == expected_run_id, "Loading models:/...@champion does not give the expected run."
    _passed("test_champion")
