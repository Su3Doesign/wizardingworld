#!/usr/bin/env bash
# The castle on the crag - open the Unreal project and build the level in one go. Edit UNREAL first.
set -euo pipefail
UNREAL="${UNREAL:-/Users/Shared/Epic Games/UE_5.6/Engine/Binaries/Mac/UnrealEditor.app/Contents/MacOS/UnrealEditor}"
HERE="$(cd "$(dirname "$0")" && pwd)"
"$UNREAL" "$HERE/WizardingWorld/WizardingWorld.uproject" -ExecutePythonScript="$HERE/WizardingWorld/Content/Python/build_world.py"
