"""MuJoCo RGB rendering helpers that do not advance simulation state."""

from __future__ import annotations

from io import BytesIO

import mujoco
import numpy as np


def render_mujoco_rgb_png(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    width: int = 640,
    height: int = 480,
) -> bytes:
    """Render current model/data to PNG without stepping or modifying state."""
    renderer = mujoco.Renderer(model, height=int(height), width=int(width))
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = np.asarray(model.stat.center, dtype=float)
    camera.distance = max(float(model.stat.extent) * 1.8, 0.5)
    camera.azimuth = 135.0
    camera.elevation = -25.0
    try:
        renderer.update_scene(data, camera=camera)
        rgb = renderer.render()
    finally:
        renderer.close()

    from PIL import Image

    output = BytesIO()
    Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB").save(output, format="PNG")
    return output.getvalue()
