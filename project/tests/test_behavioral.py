import pytest
import utils


# Tests that slight changes do not affect the output
@pytest.mark.parametrize(
    "input_a, input_b, label",
    [
        (
            "This drug is effective in treating my condition.",
            "This medication is effective in treating my condition.",
            "positive",
        ),
    ],
)
# Tests that slight changes do not affect the output
def test_invariance(input_a, input_b, label):
    """INVariance via minor wording changes (should not affect outputs)."""
    label_a = utils.get_label(text=input_a)
    label_b = utils.get_label(text=input_b)
    assert label_a == label_b == label


@pytest.mark.parametrize(
    "input, label",
    [
        (
            "This drug has improved my health significantly.",
            "positive",
        ),
        (
            "This drug made my symptoms worse.",
            "negative",
        )
    ],
)
#Tests that specific changes in sentiment lead to the expected changes in output
def test_directional(input, label):
    """DIRectional expectations (changes with known outputs)."""
    prediction = utils.get_label(text=input)
    assert label == prediction


@pytest.mark.parametrize(
    "input, label",
    [
        (
            "This drug was very helpful in reducing my symptoms.",
            "positive",
        ),
    ],
)
# Ensuring that the model returns expected labels for inputs.
def test_mft(input, label):
    """Minimum Functionality Tests (simple input/output pairs)."""
    prediction = utils.get_label(text=input)
    assert label == prediction
