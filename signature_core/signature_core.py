import os
import sys

import signature_core_cpp as _cpp
from signature_core_cpp import *
import numpy as np

def is_signature(tensor_list):
	dim = None
	for i in range(len(tensor_list)):
		if len(tensor_list[i].shape) != i:
			return False
		if dim is None and i != 0:
			dim = tensor_list[i].shape[0]
		if dim is not None and tensor_list[i].shape != (dim,)*i:
			return False
	return True

def from_numpy(tensor_list: list[np.ndarray], dim=None):
	assert is_signature(tensor_list), "tensor_list is not a signature"

	sig_dim = None

	if len(tensor_list) >= 2:
		sig_dim = tensor_list[1].shape[0]
		assert dim is None or dim == sig_dim, "dimension given in the parameters is inconsistent with the dimension of tensor_list"
	else:
		sig_dim = dim
		assert dim is not None, "could not deduce signature dimension"
	
	assert sig_dim == 2, "C++ API only supports signatures of dimension 2"

	out_sig = _cpp.Signature(order=(len(tensor_list)-1), fill_value=(0.0+0.0j))

	for ti in range(len(tensor_list)):
		tensor = tensor_list[ti].astype(np.complex128)

		for idx in np.ndindex(tensor.shape):
			out_sig.set_element( list(idx), tensor[idx] )

	return out_sig