from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import SummarizationMetric
from CohereModel import CohereModel
from a_secret_key import api_key_cohere
from src.minecraft_rag_lore import ask_lore, read_lore_book

model_cohere = CohereModel(api_key_cohere)


def test_answer_relevancy():
    input_ = read_lore_book("src/JACK THE LAW.txt")
    actual_output = ask_lore("Racconta in 10 frasi cosa è successo alla sentenza del pescatore")

    summarization_metric = SummarizationMetric(
        threshold=0.5,
        assessment_questions=[
            "viene specificato che il processo è avvenuto ad un tribunale?",
            "L'accusato ha infranto delle regole?",
            "Qualcuno è stato dichiarato colpevole?",
        ],
        model=model_cohere
    )
    test_case = LLMTestCase(
        input=input_,
        actual_output=actual_output
    )
    assert_test(test_case, [summarization_metric])
