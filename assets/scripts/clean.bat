cd libonvif
if exist build\ (
    rmdir /s /q build
)
if exist libonvif.egg-info\ (
    rmdir /s /q libonvif.egg-info
)
cd ../libavio
if exist build\ (
    rmdir /s /q build
)
if exist avio.egg-info\ (
    rmdir /s /q avio.egg-info
)
cd ../onvif-gui
if exist build\ (
    rmdir /s /q build
)
if exist onvif_gui.egg-info\ (
    rmdir /s /q onvif_gui.egg-info
)
cd ..
