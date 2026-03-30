import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageOps

try:
    from scipy.ndimage import binary_dilation
except ImportError:
    binary_dilation = None


def _pil_resampling(name, fallback):
    resampling = getattr(Image, "Resampling", None)
    if resampling is not None:
        return getattr(resampling, name)
    return fallback


PIL_BILINEAR = _pil_resampling("BILINEAR", Image.BILINEAR)


@dataclass(frozen=True)
class TopdownPose:
    x: float
    y: float
    yaw: float
    pitch: float
    camera_height: float


@dataclass(frozen=True)
class RenderBounds:
    min_x: float
    max_x: float
    min_y: float
    max_y: float


@dataclass(frozen=True)
class AlfredTopdownConfig:
    output_size: Optional[Tuple[int, int]] = (600, 600)
    fov: float = 60.0
    min_depth: float = 0.6
    max_depth: float = 10.0
    min_height: float = -0.5
    max_height: float = 15.0
    resolution: float = 0.015
    padding: float = 0.25
    display_scale: int = 3
    sample_stride: int = 1
    occupancy_threshold: float = 1.0
    occupancy_dilation: int = 1
    square_bounds: bool = False
    fixed_world_extent: float = 10.0


class AlfredTopdownBuilder:
    def __init__(self, config=None, initial_bounds=None):
        self.config = config or AlfredTopdownConfig()
        self._initial_bounds = initial_bounds
        self.reset()

    def reset(self):
        self._poses = []
        self._bounds = None
        self._counts = None

    def add_event(self, event):
        pose = self._pose_from_event(event)
        if pose is None:
            return self._render_blank()

        self._poses.append(pose)

        depth_m = self._depth_frame_to_meters(getattr(event, "depth_frame", None))
        points = self._depth_to_world_points(depth_m, pose) if depth_m is not None else np.empty((0, 2), dtype=np.float32)

        self._ensure_bounds(pose)
        self._rasterize_points(points)

        occupancy = self._counts >= self.config.occupancy_threshold
        if self.config.occupancy_dilation > 0 and binary_dilation is not None:
            occupancy = binary_dilation(occupancy, iterations=self.config.occupancy_dilation)

        image = self._render_map_image(occupancy, self._bounds, pose)
        return np.array(image)

    def _ensure_bounds(self, pose):
        """Initialise the grid on the first frame, using pre-computed bounds or a fixed extent."""
        if self._bounds is not None:
            return
        if self._initial_bounds is not None:
            self._bounds = self._initial_bounds
        else:
            half = self.config.fixed_world_extent / 2.0
            self._bounds = RenderBounds(
                min_x=pose.x - half, max_x=pose.x + half,
                min_y=pose.y - half, max_y=pose.y + half,
            )
        self._counts = self._alloc_counts(self._bounds)

    def _rasterize_points(self, points):
        if not len(points):
            return
        res = self.config.resolution
        h, w = self._counts.shape
        px = np.rint((points[:, 0] - self._bounds.min_x) / res).astype(np.int32)
        py = np.rint((self._bounds.max_y - points[:, 1]) / res).astype(np.int32)
        valid = (px >= 0) & (px < w) & (py >= 0) & (py < h)
        np.add.at(self._counts, (py[valid], px[valid]), 1.0)

    def _alloc_counts(self, bounds):
        res = self.config.resolution
        w = int(math.ceil((bounds.max_x - bounds.min_x) / res)) + 1
        h = int(math.ceil((bounds.max_y - bounds.min_y) / res)) + 1
        return np.zeros((h, w), dtype=np.float32)

    def _pose_from_event(self, event):
        metadata = getattr(event, "metadata", None)
        if not isinstance(metadata, dict):
            return None
        agent = metadata.get("agent")
        if not isinstance(agent, dict):
            return None
        position = agent.get("position")
        rotation = agent.get("rotation")
        if not isinstance(position, dict) or not isinstance(rotation, dict):
            return None

        try:
            return TopdownPose(
                x=float(position["z"]),
                y=-float(position["x"]),
                yaw=-math.radians(float(rotation["y"])),
                pitch=-math.radians(float(agent.get("cameraHorizon", 0.0))),
                camera_height=float(position["y"]),
            )
        except (KeyError, TypeError, ValueError):
            return None

    def _depth_frame_to_meters(self, depth_frame):
        if depth_frame is None:
            return None

        depth = np.asarray(depth_frame, dtype=np.float32)
        if depth.ndim == 3:
            depth = depth[..., 0]
        if depth.size == 0:
            return None

        finite = np.isfinite(depth)
        if not np.any(finite):
            return None

        max_depth = float(np.nanmax(depth[finite]))
        if max_depth > 100.0:
            depth = depth / 1000.0
        return depth

    def _depth_to_world_points(self, depth_m, pose):
        config = self.config
        height, width = depth_m.shape
        step = max(1, config.sample_stride)
        depth = depth_m[::step, ::step].astype(np.float32)

        rows = np.arange(0, height, step, dtype=np.float32)
        cols = np.arange(0, width, step, dtype=np.float32)
        vv, uu = np.meshgrid(rows, cols, indexing="ij")
        z_cam = depth

        valid = np.isfinite(z_cam) & (z_cam > config.min_depth) & (z_cam <= config.max_depth)
        if not np.any(valid):
            return np.empty((0, 2), dtype=np.float32)

        uu = uu[valid]
        vv = vv[valid]
        z_cam = z_cam[valid]

        fx = width / (2.0 * math.tan(math.radians(config.fov) / 2.0))
        fy = fx
        cx = width / 2.0
        cy = height / 2.0
        x_cam = (uu - cx) * z_cam / fx
        y_cam = (cy - vv) * z_cam / fy
        xyz_cam = np.stack([x_cam, y_cam, z_cam], axis=1)

        cos_pitch = math.cos(pose.pitch)
        sin_pitch = math.sin(pose.pitch)
        x_rotation = np.array(
            [[1.0, 0.0, 0.0], [0.0, cos_pitch, -sin_pitch], [0.0, sin_pitch, cos_pitch]],
            dtype=np.float32,
        )
        cos_yaw = math.cos(pose.yaw)
        sin_yaw = math.sin(pose.yaw)
        y_rotation = np.array(
            [[cos_yaw, 0.0, sin_yaw], [0.0, 1.0, 0.0], [-sin_yaw, 0.0, cos_yaw]],
            dtype=np.float32,
        )
        world_to_camera = x_rotation @ y_rotation
        camera_to_world = np.linalg.inv(world_to_camera).astype(np.float32)
        xyz_world = xyz_cam @ camera_to_world.T

        thor_x = -pose.y
        thor_z = pose.x
        xzy_world = xyz_world[:, [0, 2, 1]]
        xzy_world[:, 0] += thor_x
        xzy_world[:, 1] += thor_z
        xzy_world[:, 2] += pose.camera_height

        points_world = np.empty_like(xzy_world)
        points_world[:, 0] = xzy_world[:, 1]
        points_world[:, 1] = -xzy_world[:, 0]
        points_world[:, 2] = xzy_world[:, 2]

        valid_height = (points_world[:, 2] >= config.min_height) & (points_world[:, 2] <= config.max_height)
        if not np.any(valid_height):
            return np.empty((0, 2), dtype=np.float32)

        return points_world[valid_height, :2].astype(np.float32)

    def _render_map_image(self, occupancy, bounds, pose):
        base = np.where(occupancy, 255, 0).astype(np.uint8)
        image = Image.fromarray(base, mode="L").convert("RGB")
        if self.config.display_scale > 1:
            image = image.resize(
                (image.width * self.config.display_scale, image.height * self.config.display_scale),
                resample=PIL_BILINEAR,
            )

        # Draw arrow on the display_scale image BEFORE letterboxing,
        # matching the offline alfred_topdown.py render_debug_map logic.
        draw = ImageDraw.Draw(image)
        self._draw_pose_overlay(draw, pose, bounds, image.size)

        if self.config.output_size is None:
            return image

        fitted = ImageOps.contain(image, self.config.output_size, method=PIL_BILINEAR)
        canvas = Image.new("RGB", self.config.output_size, (0, 0, 0))
        offset = ((canvas.width - fitted.width) // 2, (canvas.height - fitted.height) // 2)
        canvas.paste(fitted, offset)
        return canvas

    def _draw_pose_overlay(self, draw, pose, bounds, panel_size):
        panel_w, panel_h = panel_size
        grid_w = int(math.ceil((bounds.max_x - bounds.min_x) / self.config.resolution)) + 1
        grid_h = int(math.ceil((bounds.max_y - bounds.min_y) / self.config.resolution)) + 1
        if grid_w <= 1 or grid_h <= 1:
            return

        map_x = int(round((pose.x - bounds.min_x) / self.config.resolution))
        map_y = int(round((bounds.max_y - pose.y) / self.config.resolution))
        x = map_x / max(grid_w - 1, 1) * (panel_w - 1)
        y = map_y / max(grid_h - 1, 1) * (panel_h - 1)

        line_width = max(4, self.config.display_scale)
        arrow_len = max(32.0, line_width * 10.0, max(panel_w, panel_h) * 0.055)
        forward_x = math.cos(pose.yaw)
        forward_y = -math.sin(pose.yaw)
        side_x = -forward_y
        side_y = forward_x

        tip_x = x + forward_x * arrow_len * 0.60
        tip_y = y + forward_y * arrow_len * 0.60
        left_x = x - forward_x * arrow_len * 0.30 - side_x * arrow_len * 0.38
        left_y = y - forward_y * arrow_len * 0.30 - side_y * arrow_len * 0.38
        notch_x = x - forward_x * arrow_len * 0.10
        notch_y = y - forward_y * arrow_len * 0.10
        right_x = x - forward_x * arrow_len * 0.30 + side_x * arrow_len * 0.38
        right_y = y - forward_y * arrow_len * 0.30 + side_y * arrow_len * 0.38

        draw.polygon(
            [(tip_x, tip_y), (left_x, left_y), (notch_x, notch_y), (right_x, right_y)],
            fill=(38, 136, 255),
        )

    def _render_blank(self):
        if self.config.output_size is None:
            return np.array(Image.new("RGB", (600, 600), (0, 0, 0)))
        return np.array(Image.new("RGB", self.config.output_size, (0, 0, 0)))
