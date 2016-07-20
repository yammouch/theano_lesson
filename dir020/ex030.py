import theano
from theano import In
from theano import function

x, y = theano.tensor.dscalars('x', 'y')
z = x + y
f = function([x, In(y, value=1)], z)
print f(33)
print f(33, 2)
