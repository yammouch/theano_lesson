import theano
import theano.tensor as T

a, b = T.dmatrices('a', 'b')
diff = a - b
abs_diff = abs(diff)
diff_squared = diff**2
f = theano.function([a, b], [diff, abs_diff, diff_squared])

print f([[1, 1], [1, 1]], [[0, 1], [2, 3]])
