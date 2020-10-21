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

class Transition:
    """Represents the transition between two consecutive candidates."""

    def __init__(self, from_candidate, to_candidate):
        self.from_candidate = from_candidate
        self.to_candidate = to_candidate
    
    def hash_code(self):
        return hash(self)
    
    def __hash__(self):
        return hash((self.from_candidate, self.to_candidate))
    
    def __eq__(self, other):
        return (self.from_candidate, self.to_candidate) == (other.from_candidate, other.to_candidate)
    
    def equals(self, obj):
        return (self.from_candidate == obj.from_candidate and
                self.to_candidate == obj.to_candidate)

    def __str__(self):
        return f'Transition(self.from_candidate, self.to_candidate)'
