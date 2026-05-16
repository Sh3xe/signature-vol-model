cd signature_core
cmake -B build
cd build
make
cd ../..
cp ./signature_core/build/signature_core_cpp.cpython-310-x86_64-linux-gnu.so ./signature_core_cpp.so