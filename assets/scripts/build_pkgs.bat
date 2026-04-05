pip install build
call assets\scripts\components\windows\env_variables
if not exist dist/ (
    mkdir dist
)
cd libonvif
rmdir /q /s build
python -m build
for /f %%F in ('dir /b /a-d dist\*whl') do (
    pip install dist\%%F
)
move dist\* ..\dist
cd ..
cd libavio
rmdir /q /s build
set SOURCE_DIR=%CD%
python -m build
for /f %%F in ('dir /b /a-d dist\*whl') do (
    pip install dist\%%F
)
move dist\* ..\dist
cd ..
cd kankakee
rmdir /q /s build
python -m build
for /f %%F in ('dir /b /a-d dist\*whl') do (
    pip install dist\%%F
)
move dist\* ..\dist
cd ..
cd onvif-gui
python -m build
for /f %%F in ('dir /b /a-d dist\*whl') do (
    pip install dist\%%F
)
move dist\* ..\dist
cd ..
