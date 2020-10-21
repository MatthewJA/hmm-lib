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

import copy

import numpy as np

import transition
import utils

class Step:
    """Internal state of each time step."""
    def __init__(self, candidates: set, emission_probabilities: dict,
                 transition_probabilities: dict, forward_probabilities: dict,
                 scaling_divisor: float):
        self.candidates = candidates
        self.emission_probabilities = emission_probabilities
        self.transition_probabilities = transition_probabilities
        self.forward_probabilities = forward_probabilities
        self.scaling_divisor = scaling_divisor


class ForwardBackwardAlgorithm:
    """Computes the forward-backward algorithm, also known as smoothing.
    This algorithm computes the probability of each state candidate at each time step
    given the entire observation sequence.
    """
    DELTA = 1e-6
    
    def __init__(self):
        self.steps = None
        self.prev_candidates = None
    
    def start_with_initial_state_probabilities(
            self, initial_states: set, initial_probabilities: dict):
        """Lets the computation start with the given intiial state probabilities."""
        if not self.sums_to_one(initial_probabilities.values()):
            raise ValueError('Initial state probabilities must sum to 1, summed to {}.'.format(
                sum(initial_probabilities.values())))
        
        self.initialise_state_probabilities(None, initial_states, initial_probabilities)
    
    def start_with_initial_observation(self, observation, candidates: set,
                                       emission_probabilities: dict):
        """Lets the computation start at the given first observation."""
        self.initialise_state_probabilities(observation, candidates, emission_probabilities)
        
    def next_step(self, observation, candidates: set,
                  emission_probabilities: dict,
                  transition_probabilities: dict):
        if self.steps is None:
            raise RuntimeError('start_with_initial_state_probabilities or start_with_initial_observation must be called first.')
        
        # Make defensive candidates.
        candidates = copy.copy(candidates)
        emission_probabilities = copy.copy(emission_probabilities)
        transition_probabilities = copy.copy(transition_probabilities)
        
        # On-the-fly computation of forward probabilities at each step allows
        # to efficiently (re)compute smoothing probabilities at any time step.
        prev_forward_probabilities = self.steps[-1].forward_probabilities
        cur_forward_probabilities = {}
        sum_ = 0
        for cur_state in candidates:
            forward_probability = self.compute_forward_probability(
                cur_state, prev_forward_probabilities, emission_probabilities,
                transition_probabilities)
            cur_forward_probabilities[cur_state] = forward_probability
            sum_ += forward_probability
        
        self.normalise_forward_probabilities(cur_forward_probabilities, sum_)
        self.steps.append(Step(candidates, emission_probabilities, transition_probabilities,
                          cur_forward_probabilities, sum_))
        
        self.prev_candidates = candidates
    
    def compute_smoothing_probabilities(self):
        raise NotImplementedError()
    
    def forward_probability(self, t: int, candidate):
        """Returns the probability of the specified candidate at the specified
        zero-based time step given the observations up to t."""
        if self.steps is None:
            raise RuntimeError('No time steps yet.')
        
        return self.steps[t].forward_probabilities[candidate]
    
    def current_forward_probability(self, candidate):
        """Returns the probability of the specified candidate given all previous observations."""
        if self.steps is None:
            raise RuntimeError('No time steps yet.')
        
        return self.forward_probability(self.steps[-1], candidate)
    
    def observation_log_probability(self):
        """Returns the log probability of the entire observation sequence.
        The log is returned to prevent arithmetic underflows for very small probabilities.
        """
        if self.steps is None:
            raise RuntimeError('No time steps yet.')
        
        result = 0
        for step in steps:
            result += np.log(step.scaling_divisor)
        
        return result;
    
    def compute_smoothing_probabilities(self, *args) -> list:
        """Returns the probability for all candidates of all time steps given all observations.
        The time steps include the initial states/observations time step.
        """
        if len(args) == 0:
            return self.compute_smoothing_probabilities(None)
        
        out_backward_probabilities, = args
        
        result = []
        
        list_iterator_index = len(self.steps) - 1
        if list_iterator_index == -1:
            return result;
        
        # Initial step
        step = self.steps[list_iterator_index]
        backward_probabilities = {}
        
        for candidate in step.candidates:
            backward_probabilities[candidate] = 1
        
        if out_backward_probabilities is not None:
            out_backward_probabilities.append(backward_probabilities)
        
        result.append(self.compute_smoothing_probabilities_vector(step.candidates, step.forward_probabilities, backward_probabilities))
        
        # Remaining steps
        while list_iterator_index > 0:
            next_step = step;
            list_iterator_index -= 1
            step = self.steps[list_iterator_index]
            next_backward_probabilities = backward_probabilities
            backward_probabilities = {}
            for candidate in step.candidates:
                # Using the scaling divisors of the next steps eliminates the need to
                # normalize the smoothing probabilities,
                # see also https://en.wikipedia.org/wiki/Forward%E2%80%93backward_algorithm.
                probability = self.compute_unscaled_backward_probability(candidate, next_backward_probabilities, next_step) / next_step.scaling_divisor
                backward_probabilities[candidate] = probability
            
            if out_backward_probabilities is not None:
                out_backward_probabilities.append(backward_probabilities)
            
            result.append(self.compute_smoothing_probabilities_vector(step.candidates, step.forward_probabilities, backward_probabilities))

        result.reverse()
        return result
    
    def compute_smoothing_probabilities_vector(self, candidates: set, forward_probabilities: dict, backward_probabilities: dict) -> dict:
        assert len(forward_probabilities) == len(backward_probabilities)
        result = {}
        for state in candidates:
            probability = forward_probabilities[state] * backward_probabilities[state]
            assert utils.probability_in_range(probability, self.DELTA)
            result[state] = probability
        assert self.sums_to_one(result.values())
        return result
    
    def compute_unscaled_backward_probability(self, candidate, next_backward_probabilities: dict, next_step: Step):
        result = 0
        for next_candidate in next_step.candidates:
            result += next_step.emission_probabilities[next_candidate] *\
                next_backward_probabilities[next_candidate] * self.transition_probability(
                candidate, next_candidate, next_step.transition_probabilities)
        return result
    
    def sums_to_one(self, probabilities: set):
        return abs(sum(probabilities) - 1) <= self.DELTA
    
    def initialise_state_probabilities(self, observation, candidates: set, initial_probabilities: dict):
        if self.steps is not None:
            raise RuntimeError('Initial probabilities have already been set.')
        
        candidates = copy.copy(candidates)  # Defensive copy
        self.steps = [];
        
        forward_probabilities = {}
        sum_ = 0
        for candidate in candidates:
            forward_probability = initial_probabilities[candidate]
            forward_probabilities[candidate] = forward_probability
            sum_ += forward_probability
        
        self.normalise_forward_probabilities(forward_probabilities, sum_)
        self.steps.append(Step(candidates, None, None, forward_probabilities, sum_))
        
        self.prev_candidates = candidates
        
    def compute_forward_probability(self, cur_state, prev_forward_probabilities: dict,
                                    emission_probabilities: dict, transition_probabilities: dict) -> float:
        """Returns the non-normalised forward probability of the specified state."""
        result = 0
        for prev_state in self.prev_candidates:
            result += prev_forward_probabilities[prev_state] *\
                self.transition_probability(prev_state, cur_state, transition_probabilities)
        result *= emission_probabilities[cur_state]
        return result
    
    def transition_probability(self, prev_state, cur_state, transition_probabilities: dict) -> float:
        """Returns zero probability for non-existing transitions."""
        transition_prob = transition_probabilities.get(transition.Transition(prev_state, cur_state))
        return 0 if transition_prob is None else transition_prob
    
    def normalise_forward_probabilities(self, forward_probabilities: dict, sum_: float):
        for key in forward_probabilities:
            forward_probabilities[key] = forward_probabilities[key] / sum_
