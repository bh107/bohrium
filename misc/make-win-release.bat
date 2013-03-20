@echo off
python build.py rebuild

SET INSTALL_TARGET=Bohrium-release
SET ZIPFILE=Bohrium.zip

if exist "%INSTALL_TARGET%" del /q /s "%INSTALL_TARGET%"
if exist "%ZIPFILE%" del /q "%INSTALL_TARGET%"

mkdir "%INSTALL_TARGET%"
copy "..\bridge\NumCIL\NumCIL\bin\Release\NumCIL.dll" "%INSTALL_TARGET%"
copy "..\bridge\NumCIL\NumCIL\bin\Release\NumCIL.XML" "%INSTALL_TARGET%"
copy "..\bridge\NumCIL\NumCIL.Bohrium\bin\Release\NumCIL.Bohrium.dll" "%INSTALL_TARGET%"
copy "..\bridge\NumCIL\NumCIL.Bohrium\bin\Release\NumCIL.Bohrium.XML" "%INSTALL_TARGET%"
copy "..\bridge\NumCIL\NumCIL.Unsafe\bin\Release\NumCIL.Unsafe.dll" "%INSTALL_TARGET%"
copy "..\bridge\NumCIL\NumCIL.Unsafe\bin\Release\NumCIL.Unsafe.XML" "%INSTALL_TARGET%"
copy "..\benchmark\CIL\TesterIronPython\numcil.py" "%INSTALL_TARGET%"

copy "..\core\libbh.dll" "%INSTALL_TARGET%"
copy "..\core\libbh.lib" "%INSTALL_TARGET%"
copy "..\VEM\node\libbh_vem_node.dll" "%INSTALL_TARGET%"
copy "..\VEM\node\libbh_vem_node.lib" "%INSTALL_TARGET%"
copy "..\VE\simple\libbh_ve_simple.dll" "%INSTALL_TARGET%"
copy "..\VE\simple\libbh_ve_simple.lib" "%INSTALL_TARGET%"
copy "..\VE\score\libbh_ve_score.dll" "%INSTALL_TARGET%"
copy "..\VE\score\libbh_ve_score.lib" "%INSTALL_TARGET%"
copy "..\VE\mcore\libbh_ve_mcore.dll" "%INSTALL_TARGET%"
copy "..\VE\mcore\libbh_ve_mcore.lib" "%INSTALL_TARGET%"
copy "..\VE\gpu\libbh_ve_gpu.dll" "%INSTALL_TARGET%"
copy "..\VE\gpu\libbh_ve_gpu.lib" "%INSTALL_TARGET%"

copy "config.ini.windows-example" "%INSTALL_TARGET%"\config.ini
copy "config.ini.windows-example" "%INSTALL_TARGET%"\config.ini.example

"%PROGRAMFILES%\7-zip\7z.exe" a -r "%ZIPFILE%" "%INSTALL_TARGET%"