@echo off
python build.py rebuild

SET INSTALL_TARGET=cphvb-release
SET ZIPFILE=cphVB.zip

if exist "%INSTALL_TARGET%" del /q /s "%INSTALL_TARGET%"
if exist "%ZIPFILE%" del /q "%INSTALL_TARGET%"

mkdir "%INSTALL_TARGET%"
copy "bridge\NumCIL\NumCIL\bin\Release\NumCIL.dll" "%INSTALL_TARGET%"
copy "bridge\NumCIL\NumCIL\bin\Release\NumCIL.XML" "%INSTALL_TARGET%"
copy "bridge\NumCIL\NumCIL.cphVB\bin\Release\NumCIL.cphVB.dll" "%INSTALL_TARGET%"
copy "bridge\NumCIL\NumCIL.cphVB\bin\Release\NumCIL.cphVB.XML" "%INSTALL_TARGET%"
copy "bridge\NumCIL\NumCIL.Unsafe\bin\Release\NumCIL.Unsafe.dll" "%INSTALL_TARGET%"
copy "bridge\NumCIL\NumCIL.Unsafe\bin\Release\NumCIL.Unsafe.XML" "%INSTALL_TARGET%"

copy "core\libcphvb.dll" "%INSTALL_TARGET%"
copy "core\libcphvb.lib" "%INSTALL_TARGET%"
copy "VEM\node\libcphvb_vem_node.dll" "%INSTALL_TARGET%"
copy "VEM\node\libcphvb_vem_node.lib" "%INSTALL_TARGET%"
copy "VE\simple\libcphvb_ve_simple.dll" "%INSTALL_TARGET%"
copy "VE\simple\libcphvb_ve_simple.lib" "%INSTALL_TARGET%"
copy "VE\score\libcphvb_ve_score.dll" "%INSTALL_TARGET%"
copy "VE\score\libcphvb_ve_score.lib" "%INSTALL_TARGET%"
copy "VE\mcore\libcphvb_ve_mcore.dll" "%INSTALL_TARGET%"
copy "VE\mcore\libcphvb_ve_mcore.lib" "%INSTALL_TARGET%"
copy "VE\gpu\libcphvb_ve_gpu.dll" "%INSTALL_TARGET%"
copy "VE\gpu\libcphvb_ve_gpu.lib" "%INSTALL_TARGET%"

copy "config.ini.windows-example" "%INSTALL_TARGET%"\config.ini
copy "config.ini.windows-example" "%INSTALL_TARGET%"\config.ini.example

"%PROGRAMFILES%\7-zip\7z.exe" a -r "%ZIPFILE%" "%INSTALL_TARGET%"