__author__ = 'zfh'

import numpy as np
import theano
import theano.tensor as T

t = theano.shared(0)
s = theano.tensor.vector('v')


def rec(s, first, t):
    first = s + first
    second = s
    return (first, second), {t: t + 1}


results, updates = theano.scan(
    fn=rec,
    sequences=s,
    outputs_info=[np.float64(0), None],
    non_sequences=t)

f = theano.function([s], results, updates=updates, allow_input_downcast=True)

v = np.arange(10)

print f(v)
print t.get_value()
