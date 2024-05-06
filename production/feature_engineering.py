"""Processors for the feature engineering step of the worklow.

The step loads cleaned training data, processes the data for outliers,
missing values and any other cleaning steps based on business rules/intuition.

The trained pipeline and any artifacts are then saved to be used in
training/scoring pipelines.
"""
import logging
import os.path as op

from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scripts import *

from ta_lib.core.api import (
    get_dataframe,
    get_feature_names_from_column_transformer,
    get_package_path,
    load_dataset,
    register_processor,
    save_pipeline,
    DEFAULT_ARTIFACTS_PATH
)

from ta_lib.data_processing.api import Outlier


@register_processor("feat-engg", "transform-features")
def transform_features(context, params):
    """Transform dataset to create training datasets."""
    artifacts_folder = DEFAULT_ARTIFACTS_PATH
    train_X = load_dataset(context, 'train/housing/features')
    housing_num = train_X.drop("ocean_proximity", axis=1)
    numerical_pipeline = Pipeline([
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler())
    ])
    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]

    full_pipeline = ColumnTransformer([
            ("num", numerical_pipeline, num_attribs),
            ("cat", OneHotEncoder(), cat_attribs),
        ])

    full_pipeline.fit(train_X)

    save_pipeline(
        full_pipeline, op.abspath(op.join(artifacts_folder, "preprocessing_pipeline.joblib"))
    )
    