"""Processors for the model scoring/evaluation step of the worklow."""

import os.path as op
import pandas as pd

from ta_lib.core.api import (
    DEFAULT_ARTIFACTS_PATH,
    get_dataframe,
    get_feature_names_from_column_transformer,
    get_package_path,
    hash_object,
    load_dataset,
    load_pipeline,
    register_processor,
    save_dataset,
)
from ta_lib.regression.api import RegressionReport


@register_processor("model-eval", "score-model")
def score_model(context, params):

    ifeatures_ds = "train/housing/features"
    itarget_ds = "train/housing/target"

    # load training datasets
    train_X = load_dataset(context, ifeatures_ds)
    train_y = load_dataset(context, itarget_ds)

    ifeatures_ds = "test/housing/features"
    itarget_ds = "test/housing/target"
    output = "score/housing/output"

    artifacts_folder = DEFAULT_ARTIFACTS_PATH

    # load test datasets
    test_X = load_dataset(context, ifeatures_ds)
    test_y = load_dataset(context, itarget_ds)

    # load the feature pipeline and training pipelines
    features_transformer = load_pipeline(
        op.join(artifacts_folder, "preprocessing_pipeline.joblib")
    )
    model_pipeline = load_pipeline(op.join(artifacts_folder, "train_pipeline.joblib"))

    cols = (
        list(test_X.drop("ocean_proximity", axis=1).columns)
        + ["rooms_per_household", "population_per_household", "bedrooms_per_room"]
        + list(features_transformer.transformers_[1][1].get_feature_names())
    )

    # transform the train dataset
    train_X = get_dataframe(features_transformer.transform(train_X), cols)

    # transform the test dataset
    test_X = get_dataframe(features_transformer.transform(test_X), cols)

    # make a prediction
    ypred_train = model_pipeline.predict(train_X)
    ypred_test = model_pipeline.predict(test_X)

    ols_model_report = RegressionReport(
        x_train=train_X,
        y_train=train_y,
        x_test=test_X,
        y_test=test_y,
        yhat_train=ypred_train,
        yhat_test=ypred_test,
    )

    ols_model_report.get_report(
        include_shap=False, file_path="reports/ols_model_report"
    )

    # store the predictions for any further processing.
    save_dataset(context, pd.DataFrame({"ypred": ypred_test}), output)
