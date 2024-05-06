import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    """
    A custom transformer to add combined attributes to the housing dataset.

    This transformer calculates additional attributes based on the existing features
    in the housing dataset, such as the number of rooms per household and the population
    per household. Optionally, it can also add the number of bedrooms per room.

    Parameters:
    -----------
    add_bedrooms_per_room : bool, default=True
        Whether to add the bedrooms per room attribute. If True, this attribute will
        be included in the transformed dataset; if False, it will be omitted.

    Methods:
    --------
    fit(X, y=None):
        Fit the transformer to the data. This method does not perform any actual
        computation as there is no training involved. It simply returns self.

        Parameters:
        -----------
        X : array-like or DataFrame, shape (n_samples, n_features)
            The input data.

        y : array-like, shape (n_samples,) or (n_samples, n_targets), optional
            The target labels.

        Returns:
        --------
        self : object
            Returns the instance itself.
    """

    def __init__(self, add_bedrooms_per_room=True):  # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X, y=None):
        rooms_per_household = X.iloc[:, rooms_ix] / X.iloc[:, households_ix]
        population_per_household = X.iloc[:, population_ix] / X.iloc[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X.iloc[:, bedrooms_ix] / X.iloc[:, rooms_ix]
            return np.c_[
                X, rooms_per_household, population_per_household, bedrooms_per_room
            ]

        else:
            return np.c_[X, rooms_per_household, population_per_household]

    def inverse_transform(self, X, y=None):
        if self.add_bedrooms_per_room:
            return X[:, :-3]
        else:
            return X[:, :-2]
