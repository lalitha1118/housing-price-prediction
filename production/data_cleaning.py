"""Processors for the data cleaning step of the worklow.

The processors in this step, apply the various cleaning steps identified
during EDA to create the training datasets.
"""

import numpy as np
import os
import pandas as pd
import urllib
from scripts import *
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit

from ta_lib.core.api import (
    custom_train_test_split,
    load_dataset,
    register_processor,
    save_dataset,
)


@register_processor("data-cleaning", "housingds")
def clean_housing_table(context, params):
    ds = "raw/housing"
    house_df = load_dataset(context, ds)
    median = house_df["total_bedrooms"].median()
    house_df["total_bedrooms"].fillna(median, inplace=True)
    save_dataset(context, house_df, "cleaned/housing")
    return house_df


@register_processor("data-cleaning", "train-test")
def create_training_datasets(context, params):

    house_df = load_dataset(context, "cleaned/housing")
    housing_num = house_df.drop("ocean_proximity", axis=1)
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(housing_num)
    housing_num = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)
    housing_categorical = house_df[["ocean_proximity"]]
    house_prepared = housing_num.join(housing_categorical)
    save_dataset(context, house_prepared, "processed/housing")
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=params["test_size"], random_state=context.random_seed
    )
    house_df_train, house_df_test = custom_train_test_split(
        house_prepared, splitter, by=binned_median_income
    )
    train_X, train_y = house_df_train.get_features_targets(
        target_column_names=params["target"]
    )
    save_dataset(context, train_X, "train/housing/features")
    save_dataset(context, train_y, "train/housing/target")
    test_X, test_y = house_df_test.get_features_targets(
        target_column_names=params["target"]
    )
    save_dataset(context, test_X, "test/housing/features")
    save_dataset(context, test_y, "test/housing/target")
