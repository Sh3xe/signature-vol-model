import numpy as np
import numpy.random as npr
from functools import cache

def is_signature(tensor_list):
	dim = None
	for i in range(len(tensor_list)):
		if len(tensor_list[i].shape) != i:
			return False
		if dim == None and i != 0:
			dim = tensor_list[i].shape[0]
		if dim != None and tensor_list[i].shape != (dim,)*i:
			return False
	return True

class Sig:
	def __init__(self, tensor_list: list[np.ndarray], dim=None):
		assert is_signature(tensor_list), "tensor_list is not a signature"

		if len(tensor_list) >= 2:
			self.dim = tensor_list[1].shape[0]
			assert dim == None or dim == self.dim, "dimension given in the parameters is inconsistent with the dimension of tensor_list"
		else:
			self.dim = dim
			assert dim is not None, "could not deduce signature dimension"
		
		self.data = tensor_list

	def __add__(self, other):
		assert self.dim == other.dim, "cannot add two signature of different signatures"
		
		added_sigs = []
		for i in range(max(len(self.data), len(other.data))):
			if i >= len(self.data):
				added_sigs.append( other.data[i] )
			elif i >= len(other.data):
				added_sigs.append( self.data[i] )
			else:
				added_sigs.append( self.data[i] + other.data[i] )

		# scalar np.array sums to scalar, we need to convert it back to array
		added_sigs[0] = np.array(added_sigs[0])
		return added_sigs

	def __iadd__(self, other):
		assert self.dim == other.dim, "cannot add two signature of different signatures"
		
		# Extend in-place if necessary
		if len(other.data) > len(self.data):
			self.data.extend( [ np.zeros( (self.dim,)*order ) for order in range(len(self.data), len(other.data)+1) ] )
		
		# Sum ...
		for i in range(len(other.data)):
			self.data[i] += other.data[i]

		return self

	def __mul__(self, other, trunc=None):
		assert self.dim == other.dim, "cannot add two signature of different signatures"

		if trunc == None:
			trunc = len(self.data) + len(other.data)

		sig_prod = [ np.zeros(tuple(self.dim for _ in range(n))) for n in range(trunc) ]

		for i in range(len(self.data)):
			for j in range(len(other.data)):
				if i+j > trunc:
					continue
				sig_prod[i+j] += np.tensordot(self.data[i], other.data[j], axes=0)
		
		return Sig(sig_prod)

	def order(self):
		return len(self.data)

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		assert index >= 0, "Index of a signature must be positive"
		return self.data[index] if index < len(self.data) else np.zeros( (self.dim,)*index )

	def shuffle(self, other, trunc=None):
		assert self.dim == other.dim, "cannot add two signature of different signatures"

		if trunc == None:
			trunc = len(self.data) + len(other.data)
		
		shuffle_sig = [ np.zeros( tuple(self.dim for _ in range(n)) ) for n in range(trunc+1) ]

		for i in range(len(self.data)):
			for j in range(len(other.data)):
				tensor_order = len(self.data[i].shape) + len(other.data[j].shape)
				if tensor_order > trunc:
					continue
				prod_shape = tuple(self.dim for _ in range(tensor_order))
				
				# the products will all be a tensor of the same dim and order
				for self_index in np.ndindex(self.data[i].shape):
					for other_index in np.ndindex(other.data[j].shape):
						el = self.data[i][self_index] * other.data[j][other_index] * shuffle_product_basis(tuple(self_index), tuple(other_index), self.dim)
						shuffle_sig[len(prod_shape)] += el

		return Sig(shuffle_sig)

@cache
def make_canonical_element(indices, n):
	x = np.zeros([n for _ in range(len(indices))])
	x[tuple(indices)] = 1
	return x

@cache
def shuffle_product_basis(left: tuple[int], right: tuple[int], n: int) -> list[np.ndarray]:
	"""
	Perform the shuffle product between two signature basis

	Args:
		left, right: "word", i.e. any element of the canonical basis of any tensor of any order, with dimension equal to n.
		n: dimension of the tensors
	"""
	
	if len(left) == 0:
		return make_canonical_element(right, n)
	elif len(right) == 0:
		return make_canonical_element(left, n)
	
	(begin_left, last_left) = left[:-1], left[-1]
	(begin_right, last_right) = right[:-1], right[-1]

	sub_left = shuffle_product_basis(begin_left, right, n)
	sub_right = shuffle_product_basis(left, begin_right, n)

	return (
		np.tensordot(sub_left, make_canonical_element( (last_left,), n), axes=0) +
		np.tensordot(sub_right, make_canonical_element( (last_right,), n), axes=0)
	)

def scalar_prod(sig: Sig, scalar):
		out_sig = []
		for i in range(len(sig.data)):
			out_sig.append(sig.data[i] * scalar)
		return Sig(out_sig)

def shuffle_product(left: np.ndarray, right: np.ndarray) -> list[np.ndarray]:
	# Special case: scalar
	if len(left.shape) == 0 and len(right.shape)  == 0:
		return np.array(left * right)
	
	# guess the dimension of the product
	if len(left.shape) == 0:
		dim = right.shape[0]
	else:
		dim = left.shape[0]
		
	# some utility values on the tensors (dimension, trucation of output signature)
	tensor_order = len(left.shape) + len(right.shape)

	# the products will all be a tensor of the same dim and order
	product_tensor = np.zeros( tuple(dim for _ in range(tensor_order)) )
	for self_index in np.ndindex(left.shape):
		for other_index in np.ndindex(right.shape):
			el = left[self_index] * right[other_index] * shuffle_product_basis(tuple(self_index), tuple(other_index), dim)
			product_tensor += el

	return product_tensor

def signature( x: np.ndarray, trunc: int ):
	"""
	Compute the truncated signature of x, where x is a sample of some stochastic process
	trunc is the maximum tensor order of the signature
	"""
	assert trunc >= 0, "trunc is the maximum tensor order, it cannot be negative"

	n_divs, dim = x.shape[1], x.shape[0]
	sig = [np.ones(n_divs), x]

	if trunc <= 2:
		return sig[:trunc+1]

	# Compute the next signature recursively
	for n in range(2, trunc+1):
		# numpy shape corresponding to the nth tensor of the signature
		shape_n = tuple(dim for _ in range(n))
		sig_n = np.zeros( shape=shape_n+(n_divs,) )
		for index in np.ndindex(shape_n):
			mean_vals = (sig[n-1][index[:-1]][1:] + sig[n-1][index[:-1]][:-1]) / 2
			integral = mean_vals * (x[index[-1], 1:] - x[index[-1], :-1])
			sig_n[index] = np.concatenate( ([0], integral.cumsum()) )
		sig.append(sig_n)
	return sig

def bracket(sig_left, sig_right):
	bracket_sum = 0.0
	for i in range(min(len(sig_left), len(sig_right))):
		bracket_sum += (sig_left[i] * sig_right[i]).sum()
	return bracket_sum

def bracket_with_process(cst_sig, process_sig, max_order = None):
	if max_order == None:
		max_order = min(len(cst_sig), len(process_sig))
	else:
		max_order = min(len(cst_sig), len(process_sig), max_order)

	bracket_sum = np.zeros(process_sig[0].shape)
	for i in range(max_order):
		for index in np.ndindex(cst_sig[i].shape):
			bracket_sum += cst_sig[i][index] * process_sig[i][index]
	return bracket_sum