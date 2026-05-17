import signature_core_cpp as sigcpp

sig_cpp = sigcpp.Signature()

sig_cpp.set_data(1054)

print(sig_cpp.get_data())
other_sig = sig_cpp
other_sig.set_data(1)
print(sig_cpp.get_data())
