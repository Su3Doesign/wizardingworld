"""Render the shots that build_world.py queued in Movie Render Queue (local, inside the editor).

    Tools > Execute Python Script...  and pick this file                      -> renders every queued shot
    or, in the Output Log (Python):  import render_shots; render_shots.run(["CAM_01_Gorge_Mist", "CAM_02_Boats_Night"])

Every shot's Level Sequence carries its own lighting preset (sun / moon, fog, mist banks, lanterns, window glow), so the
whole queue renders in one go.  Images: <Project>/Saved/MovieRenders/<shot>/LS_<shot>.<frame>.png
Single-frame stills: the sequences of the static shots are one frame long; the animated ones (gorge drift, boats,
viaduct walk, Quidditch flyover, grand orbit) render every frame - untick 'Render' on a job to skip it.
"""
import unreal


def run(names=None):
    sub = unreal.get_editor_subsystem(unreal.MoviePipelineQueueSubsystem)
    queue = sub.get_queue()
    jobs = list(queue.get_jobs())
    if not jobs:
        raise RuntimeError("The Movie Render Queue is empty - run build_world.run() first.")
    n = 0
    for j in jobs:
        name = str(j.get_editor_property("job_name"))
        on = names is None or name in names
        try:
            j.set_is_enabled(on)
        except Exception:                                          # noqa: BLE001 - older engines
            j.set_editor_property("enabled", on)
        n += int(on)
    unreal.log(f"[WW] rendering {n} of {len(jobs)} shots with Movie Render Queue ...")
    sub.render_queue_with_executor(unreal.MoviePipelinePIEExecutor)


if __name__ == "__main__":
    run()
