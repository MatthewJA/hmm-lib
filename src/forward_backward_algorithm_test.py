"""
Copyright (C) 2015, BMW Car IT GmbH
Copyright (C) 2020, Geoscience Australia

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

This file is a derivative work from the original software by
Stefan Holder (BMW), as it is a direct port to Python.
"""

from enum import Enum
import unittest

import transition
import forward_backward_algorithm

class Rain(Enum):
    T = True
    F = False
    
    def __str__(self):
        return ['Sun', 'Rain'][self.value]


class Umbrella(Enum):
    T = True
    F = False
    
    def __str__(self):
        return ['No umbrella', 'Umbrella'][self.value]

    
class TestForwardBackwardAlgorithm(unittest.TestCase):
    
    def test_forward_backward(self):
        """Example taken from https://en.wikipedia.org/wiki/Forward%E2%80%93backward_algorithm."""
        candidates = [Rain.T, Rain.F]
        
        initial_state_probabilities = {
            Rain.T: 0.5,
            Rain.F: 0.5,
        }
        
        emission_probabilities_for_umbrella = {
            Rain.T: 0.9,
            Rain.F: 0.2,
        }
        
        emission_probabilities_for_no_umbrella = {
            Rain.T: 0.1,
            Rain.F: 0.8,
        }
        
        transition_probabilities = {
            transition.Transition(Rain.T, Rain.T): 0.7,
            transition.Transition(Rain.T, Rain.F): 0.3,
            transition.Transition(Rain.F, Rain.T): 0.3,
            transition.Transition(Rain.F, Rain.F): 0.7,
        }
        
        fw = forward_backward_algorithm.ForwardBackwardAlgorithm()
        fw.start_with_initial_state_probabilities(candidates, initial_state_probabilities)
        fw.next_step(Umbrella.T, candidates, emission_probabilities_for_umbrella,
                transition_probabilities)
        fw.next_step(Umbrella.T, candidates, emission_probabilities_for_umbrella,
                transition_probabilities)
        fw.next_step(Umbrella.F, candidates, emission_probabilities_for_no_umbrella,
                transition_probabilities)
        fw.next_step(Umbrella.T, candidates, emission_probabilities_for_umbrella,
                transition_probabilities)
        fw.next_step(Umbrella.T, candidates, emission_probabilities_for_umbrella,
                transition_probabilities)
        
        result = fw.compute_smoothing_probabilities()
        self.assertEqual(len(result), 6)

        DELTA = 1e-4
        self.assertTrue(abs(result[0][Rain.T] - 0.6469) <= DELTA)
        self.assertTrue(abs(result[0][Rain.F] - 0.3531) <= DELTA)
        self.assertTrue(abs(result[1][Rain.T] - 0.8673) <= DELTA)
        self.assertTrue(abs(result[1][Rain.F] - 0.1327) <= DELTA)
        self.assertTrue(abs(result[2][Rain.T] - 0.8204) <= DELTA)
        self.assertTrue(abs(result[2][Rain.F] - 0.1796) <= DELTA)
        self.assertTrue(abs(result[3][Rain.T] - 0.3075) <= DELTA)
        self.assertTrue(abs(result[3][Rain.F] - 0.6925) <= DELTA)
        self.assertTrue(abs(result[4][Rain.T] - 0.8204) <= DELTA)
        self.assertTrue(abs(result[4][Rain.F] - 0.1796) <= DELTA)
        self.assertTrue(abs(result[5][Rain.T] - 0.8673) <= DELTA)
        self.assertTrue(abs(result[5][Rain.F] - 0.1327) <= DELTA)

        
if __name__ == '__main__':
    unittest.main()
