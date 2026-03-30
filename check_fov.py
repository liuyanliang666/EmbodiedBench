"""Quick script to check AI2-THOR's actual camera FOV."""

import sys
sys.path.insert(0, ".")

from embodiedbench.envs.eb_alfred.gen import constants
from ai2thor.controller import Controller

c = Controller(quality="MediumCloseFitShadows")
c.start(
    x_display="1",
    player_screen_height=600,
    player_screen_width=600,
)
c.reset("FloorPlan1")
event = c.step(dict(
    action="Initialize",
    gridSize=constants.AGENT_STEP_SIZE / constants.RECORD_SMOOTHING_FACTOR,
    cameraY=constants.CAMERA_HEIGHT_OFFSET,
    renderImage=True,
    renderDepthImage=True,
    renderClassImage=True,
    renderObjectImage=True,
    visibility_distance=constants.VISIBILITY_DISTANCE,
    makeAgentsVisible=False,
))

m = event.metadata
print("=" * 50)
print(f"fov                = {m.get('fov')}")
print(f"screenWidth        = {m.get('screenWidth')}")
print(f"screenHeight       = {m.get('screenHeight')}")
print(f"cameraPosition     = {m.get('cameraPosition')}")
print(f"agent.cameraHorizon= {m.get('agent', {}).get('cameraHorizon')}")

# dump any key containing 'fov' or 'field' (case-insensitive)
fov_keys = {k: m[k] for k in m if "fov" in k.lower() or "field" in k.lower()}
if fov_keys:
    print(f"fov-related keys   = {fov_keys}")
else:
    print("No 'fov'/'field' key found in metadata, listing all top-level keys:")
    print(sorted(m.keys()))

print("=" * 50)
c.stop()
