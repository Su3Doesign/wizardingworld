@echo off
rem The castle on the crag - open the Unreal project and build the level in one go.
rem Edit UE to point at your engine (5.6 recommended; 5.4 - 5.7 should work).
set UE="C:\Program Files\Epic Games\UE_5.6\Engine\Binaries\Win64\UnrealEditor.exe"
set HERE=%~dp0
%UE% "%HERE%WizardingWorld\WizardingWorld.uproject" -ExecutePythonScript="%HERE%WizardingWorld\Content\Python\build_world.py"
