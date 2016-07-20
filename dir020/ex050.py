import theano.tensor as T
from theano import shared, function

state = shared(0)
inc = T.dscalar('inc')

fn_of_state = state * 2 + inc
# The type of foo must match the shared variable we are replacing
# with the ''givens''
foo = T.scalar(dtype=state.dtype)
skip_shared = function([inc, foo], fn_of_state, givens=[(state, foo)])
print skip_shared(1, 3)
print state.get_value()
