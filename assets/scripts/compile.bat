cd libonvif
rmdir /s /q build
rmdir /s /q libonvif.egg-info
pip install -v .
cd ../libavio
rmdir /s /q build
rmdir /s /q avio.egg-info
pip install -v .
cd ../onvif-gui
rmdir /s /q build
rmdir /s /q onvif_gui.egg-info
pip install .
