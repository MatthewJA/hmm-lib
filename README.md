# Overview

This library implements Hidden Markov Models (HMM) for time-inhomogeneous Markov processes.
This means that, in contrast to many other HMM implementations, there can be different
states and a different transition matrix at each time step.

This is a port of [https://github.com/bmwcarit/hmm-lib](bmwcarit/hmm-lib) to Python.

This library provides an implementation of
* The forward-backward algorithm, which computes the probability of all state candidates given
the entire sequence of observations. This process is also called smoothing.

# License

This library is licensed under the
[Apache 2.0 license](http://www.apache.org/licenses/LICENSE-2.0.html).
