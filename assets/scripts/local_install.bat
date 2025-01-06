for /R libonvif\dist %%F in (*.whl) do pip install "%%F"
for /R libavio\dist %%F in (*.whl) do pip install "%%F"
for /R kankakee\dist %%F in (*.whl) do pip install "%%F"
for /R onvif-gui\dist %%F in (*.whl) do pip install "%%F"
