import unittest

from .label import process_responses, TRIAL_COUNT

class TestProcessResponses(unittest.TestCase):

    def test_basic_functionality(self):
        trials = [{'is_mountain': False, 'responses': []} for _ in range(TRIAL_COUNT)]
        output = process_responses(trials)
        self.assertEqual(output, [None]*TRIAL_COUNT)  # No responses, all should be None
    
    def test_first_trial_responses(self):
        trials = [{'is_mountain': False, 'responses': [100]}] + [{'is_mountain': False, 'responses': []} for _ in range(TRIAL_COUNT-1)]
        output = process_responses(trials)
        self.assertEqual(output[0], 100)
    
    def test_unambiguous_correct_responses(self):
        trials = [{'is_mountain': False, 'responses': []}]
        trials += [{'is_mountain': False, 'responses': [300, 700]} for _ in range(TRIAL_COUNT-1)]
        output = process_responses(trials)
        # Only second set of responses should be taken as it is not mountain
        self.assertEqual(output, [None] + [700]*49)