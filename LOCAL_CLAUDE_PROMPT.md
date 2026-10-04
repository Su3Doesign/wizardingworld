Paste this into a Claude Code session running on the PC that has Unreal Engine, opened in this folder:

---

This repository contains a procedural world: a Gothic castle on a granite crag above a lake, with a river gorge, a
waterfall, forests and mountains. `WizardingWorld/` is a content-only Unreal Engine 5.6 project;
`WizardingWorld/Content/Python/build_world.py` builds the level from `WizardingWorld/SourceAssets` (FBX + textures +
instance records). Read `README.md` first.

Please do the following:

1. Find my Unreal Engine install (`C:\Program Files\Epic Games\UE_5.*`). Open `WizardingWorld/WizardingWorld.uproject`
   (it enables the Python, Editor Scripting, Sequencer Scripting and Movie Render Queue plugins).
2. Run the builder headless first to catch API problems quickly:
   `UnrealEditor-Cmd.exe <abs path>\WizardingWorld\WizardingWorld.uproject -ExecutePythonScript="<abs path>\WizardingWorld\Content\Python\build_world.py" -unattended -nosplash`
   Read `WizardingWorld/Saved/Logs/WizardingWorld.log` for lines tagged `[WW]` and Python errors. The script was
   verified against the UE 5.6 Python API stub and executed end to end against a mocked `unreal` module, but it has
   not run inside Unreal yet: fix any API mismatch for my engine version and re-run until the summary line reports
   0 failed steps.
3. Open the editor, load `/Game/WizardingWorld/Maps/WizardingWorld`, pilot `CAM_01_Gorge_Mist` and check: the castle
   stands on the crag (not floating, not mirrored), the trees stand on the ground, the moss sits on the rock ledges.
   If something is 100x off or mirrored, the calibration step's log line says why.
4. For each preset run `import build_world; build_world.apply_preset("<preset>")` (mist, day, sunset, dusk, night),
   take a High Resolution Screenshot from the matching camera, and tune the values in `SourceAssets/scene.json` /
   `CFG` until the mist shot looks like a misty Highland morning and the night shot like a moonlit lake.
5. Render all shots: `import render_shots; render_shots.run()`. Show me the images from `Saved/MovieRenders/`.
6. Then the realism upgrades from README "Pushing it to photoreal": Megascans granite + moss into `MI`s of
   M_Terrain / M_Boulder, Megascans conifers for the hero foreground, then re-render.
