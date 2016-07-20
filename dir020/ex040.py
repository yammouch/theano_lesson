import theano.tensor as T
from theano import shared, function

state = shared(0)
inc = T.iscalar('inc')
accumulator = function([inc], state, updates=[(state, state+inc)])

print state.get_value()
print accumulator(1)
print state.get_value()
print accumulator(300)
print state.get_value()

state.set_value(-1)
print accumulator(3)
print state.get_value()

decrementor = function([inc], state, updates=[(state, state-inc)])
print decrementor(2)
print state.get_value()
