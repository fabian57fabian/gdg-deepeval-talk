from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.models import GPTModel
from CohereModel import CohereModel
from a_secret_key import api_key_cohere, openai_api_key

model_cohere = CohereModel(api_key_cohere)
model_gpt = GPTModel(model="gpt-3.5-turbo-1106", _openai_api_key=openai_api_key)


def test_answer_relevancy():
    answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.5, model=model_cohere)
    test_case = LLMTestCase(
        input="What if these shoes don't fit?",
        actual_output="We offer a 30-day full refund at no extra cost."
    )
    assert_test(test_case, [answer_relevancy_metric])
