from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import BaseMetric


class LatencyMetric(BaseMetric):
    # This metric by default checks if the latency is greater than 10 seconds
    def __init__(self, max_seconds: int = 10):
        self.threshold = max_seconds

    def measure(self, test_case: LLMTestCase, *args, **kwargs):
        # Set self.success and self.score in the "measure" method
        self.success = test_case.latency <= self.threshold
        if self.success:
            self.score = 1
        else:
            self.score = 0

        # You can also optionally set a reason for the score returned.
        # This is particularly useful for a score computed using LLMs
        self.reason = "Too slow!"
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "Latency"


def test_answer_relevancy():
    latency_metric = LatencyMetric(max_seconds=10.0)
    test_case = LLMTestCase(input="...", actual_output="...", latency=8.3)
    assert_test(test_case, [latency_metric])
