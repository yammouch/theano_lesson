import theano

a = theano.tensor.vector()
out = a + a**10
f = theano.function([a], out)
print(f([0, 1, 2]))
