from sklearn.utils.estimator_checks import parametrize_with_checks

from ta_lib.transformers import CombinedAttributesAdder


@parametrize_with_checks([CombinedAttributesAdder()])
def test_using_check_estimator(estimator, check):
    check(estimator)
