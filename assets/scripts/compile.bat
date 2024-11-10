cd libonvif
if exist build\ (
    rmdir /s /q build
)
if exist libonvif.egg-info\ (
    rmdir /s /q libonvif.egg-info
)
pip install -v .
cd ../libavio
if exist build\ (
    rmdir /s /q build
)
if exist avio.egg-info\ (
    rmdir /s /q avio.egg-info
)
pip install -v .
cd ../liblivemedia
if exist build\ (
    rmdir /s /q build
)
if exist liblivemedia.egg-info\ (
    rmdir /s /q liblivemedia.egg-info
)
pip install -v .
cd ../kankakee
if exist build\ (
    rmdir /s /q build
)
if exist kankakee.egg-info\ (
    rmdir /s /q kankakee.egg-info
)
pip install -v .
cd ../onvif-gui
if exist build\ (
    rmdir /s /q build
)
if exist onvif_gui.egg-info\ (
    rmdir /s /q onvif_gui.egg-info
)
pip install .
cd ..
