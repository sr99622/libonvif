echo "BUILD.SH"
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX:PATH=${PREFIX} -DCMAKE_BUILD_TYPE:STRING=Release -DBUILD_GUI=ON ..
make install
