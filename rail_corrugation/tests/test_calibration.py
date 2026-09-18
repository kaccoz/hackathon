import numpy as np
from sklearn.dummy import DummyClassifier

from rail_cdm.calibration import ProbabilityAdjustedClassifier
from rail_cdm.train import make_final_model


def test_probability_multiplier_can_raise_minority_class_prediction() -> None:
    x = np.array([[0], [1], [2]])
    y = np.array(["Normal", "Normal", "Side I"])
    model = ProbabilityAdjustedClassifier(
        estimator=DummyClassifier(strategy="prior"),
        class_multipliers=(("Side I", 3.0),),
    ).fit(x, y)

    probabilities = model.predict_proba([[10]])

    assert np.isclose(probabilities.sum(), 1.0)
    assert model.predict([[10]])[0] == "Side I"


def test_final_model_keeps_smote_inside_the_training_pipeline() -> None:
    model = make_final_model()

    assert model.named_steps["sampler"].k_neighbors == 2
    assert model.named_steps["sampler"].sampling_strategy == {"Side I": 48, "Side II": 48}
    classifier = model.named_steps["model"].estimator.named_steps["classifier"]
    assert classifier.class_weight is None
