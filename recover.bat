@echo off
rem Recover Studio without the browser:
rem   recover.bat "D:\digitized\side A.wav" [--source vinyl] [--style warm] [--intensity medium] [--reference ref.wav]
rem               [--preview] [--stems --vocals 2] [--engine auto|ai|dsp] [--sr 48000] [--bits 24] [--out D:\restored]
rem Drag an audio file onto this .bat to restore it with automatic settings. Results: projects\<date>_<name>\full\
setlocal
chcp 65001 >nul
set "PYTHONNOUSERSITE=1"
set "PYTHONPATH="
set "PYTHONIOENCODING=utf-8"
if "%~1"=="" (
  set /p "IN=Перетащите сюда аудиофайл и нажмите Enter: "
) else (
  set "IN=%~1"
)
set "IN=%IN:"=%"
"%~dp0python\python.exe" -s "%~dp0app\recover.py" "%IN%" %2 %3 %4 %5 %6 %7 %8 %9
pause
