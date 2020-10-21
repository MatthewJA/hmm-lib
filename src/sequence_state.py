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

class SequenceState:
    """State of the most likely sequence with additional information."""
    
    def __init__(self, state, observation, transition_descriptor, smoothing_probability: float):
        self.state = state
        self.observation = observation
        self.transition_descriptor = transition_descriptor
        self.smoothing_probability = smoothing_probability
