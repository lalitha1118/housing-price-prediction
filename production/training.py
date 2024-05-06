"""Processors for the model training step of the worklow."""
import logging
import os.path as op

from sklearn.pipeline import Pipeline


from ta_lib.core.api import (
    get_dataframe,
    get_feature_names_from_column_transformer,
    get_package_path,
    load_dataset,
    load_pipeline,
    register_processor,
    save_pipeline,
    DEFAULT_ARTIFACTS_PATH
)
from ta_lib.regression.api import SKLStatsmodelOLS



@register_processor("model-gen", "train-model")
def train_model(context, params):

    artifacts_folder = DEFAULT_ARTIFACTS_PATH

    input_features_ds = "train/housing/features"
    input_target_ds = "train/housing/target"
    train_X = load_dataset(context, input_features_ds)
    train_y = load_dataset(context, input_target_ds)

    # load pre-trained feature pipelines and other artifacts
    full_pipeline = load_pipeline(op.join(artifacts_folder, "preprocessing_pipeline.joblib"))

    cols = list(train_X.drop("ocean_proximity", axis=1).columns) + ["rooms_per_household", "population_per_household",
                         "bedrooms_per_room"] + list(full_pipeline.transformers_[1][1].get_feature_names())

    train_X = get_dataframe(
        full_pipeline.fit_transform(train_X, train_y),
        cols
    )

    # create training pipeline
    regression_pipeln_ols = Pipeline([("estimator", SKLStatsmodelOLS())])
    regression_pipeln_ols.fit(train_X, train_y.values.ravel())

    # save fitted training pipeline
    save_pipeline(
        regression_pipeln_ols, op.abspath(op.join(artifacts_folder, "train_pipeline.joblib"))
    )