mkdir build
cd build
cmake -G "NMake Makefiles" -DCMAKE_INSTALL_PREFIX:PATH="%LIBRARY_PREFIX%" -DCMAKE_BUILD_TYPE:STRING=Release -DBUILD_GUI=ON ..
nmake
nmake install
