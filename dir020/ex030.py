import theano.tensor as T
from theano import In
from theano import function

x, y = T.dscalars('x', 'y')
z = x + y
f = function([x, In(y, value=1)], z)
print f(33)
print f(33, 2)

x, y, w = T.dscalars('x', 'y', 'w')
z = (x + y) * w
f = function([x, In(y, value=1), In(w, value=2, name='w_by_name')], z)
print f(33)
print f(33, 2)
print f(33, 0, 1)
print f(33, w_by_name=1)
print f(33, w_by_name=1, y=0)
