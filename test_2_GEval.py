from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams
from CohereModel import CohereModel
from a_secret_key import api_key_cohere

model_cohere = CohereModel(api_key_cohere)


def test_answer_relevancy():
    geval_metric = GEval(
        name="Coherence",
        criteria="Coherence - determine if the actual output is coherent with the input.",
        # NOTE: you can only provide either criteria or evaluation_steps, and not both
        evaluation_steps=["Check whether the sentences in 'actual output' aligns with that in 'input'"],
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        model=model_cohere
    )
    test_case = LLMTestCase(
        input="What if these shoes don't fit?",
        actual_output="We offer a 30-day full refund at no extra cost."
    )
    assert_test(test_case, [geval_metric])
