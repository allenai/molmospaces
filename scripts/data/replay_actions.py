import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import ffmpegcv
import h5py
import mujoco as mj
import numpy as np
import tyro
from tqdm import tqdm

from molmo_spaces.renderer.abstract_renderer import MjAbstractRenderer
from molmo_spaces.renderer.filament_rendering import MjFilamentRenderer
from molmo_spaces.renderer.opengl_rendering import MjOpenGLRenderer

EMPTY_SCENE_XML = Path(__file__).parent / "scene_empty.xml"


@dataclass
class Args:
    robot_filepath: Path
    data_filepath: Path


ROBOT_POS = [0.0, 0.0, 0.0]
ROBOT_QUAT = [1.0, 0.0, 0.0, 0.0]

HAS_FILAMENT: bool = getattr(mj, "mjRENDERER", "classic") == "filament"

RENDERER_WIDTH = 640
RENDERER_HEIGHT = 480

CTRL_TIMESTEP = 0.066


def load_dicts_data(data: h5py.Dataset) -> list[dict]:
    ret = []
    for i in range(data.shape[0]):
        d = json.loads(data[i].tobytes().decode("utf-8").rstrip("\x00"))
        ret.append(d)
    return ret


def resize_for_cv(frame: np.ndarray, width: int, height: int) -> cv2.Mat | np.ndarray:
    if frame.shape != (height, width):
        return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LANCZOS4)
    return frame


@dataclass
class VideoRecorderCfg:
    filepath: Path
    width: int
    height: int
    fps: int


class VideoRecorder:
    def __init__(self, filepath: Path, width: int, height: int, fps: int = 30) -> None:
        self.cfg = VideoRecorderCfg(filepath, 2 * width, height, fps)

        self._writer = ffmpegcv.VideoWriter(
            self.cfg.filepath.as_posix(),
            fps=self.cfg.fps,
            resize=(self.cfg.width, self.cfg.height),
        )

    def push_frame(self, frame: np.ndarray) -> None:
        frame_resized = resize_for_cv(frame, self.cfg.width, self.cfg.height)
        frame_bgr = cv2.cvtColor(frame_resized, cv2.COLOR_RGB2BGR)
        self._writer.write(frame_bgr)

    def close(self) -> None:
        self._writer.release()

    def __del__(self) -> None:
        self.close()


def main() -> int:
    args = tyro.cli(Args)

    if not args.robot_filepath.is_file():
        return 1
    if not args.data_filepath.is_file():
        return 1

    spec = mj.MjSpec.from_file(EMPTY_SCENE_XML.as_posix())
    robot_spec = mj.MjSpec.from_file(args.robot_filepath.as_posix())

    robot_frame = spec.worldbody.add_frame(pos=ROBOT_POS, quat=ROBOT_QUAT)
    robot_root = robot_spec.worldbody.first_body()
    robot_frame.attach_body(robot_root)

    model: mj.MjModel = spec.compile()
    data: mj.MjData = mj.MjData(model)

    video_suffix = "filament" if HAS_FILAMENT else "classic"
    video_filepath = Path(f"video_{video_suffix}.mp4")
    recorder = VideoRecorder(video_filepath, RENDERER_WIDTH, RENDERER_HEIGHT)

    renderer: MjAbstractRenderer | None = None
    if HAS_FILAMENT:
        renderer = MjFilamentRenderer(model=model, width=RENDERER_WIDTH, height=RENDERER_HEIGHT)
    else:
        renderer = MjOpenGLRenderer(model=model, width=RENDERER_WIDTH, height=RENDERER_HEIGHT)

    fhandle = h5py.File(args.data_filepath.as_posix())
    actions = load_dicts_data(fhandle["traj_0/actions/commanded_action"])  # type: ignore

    mj.mj_resetData(model, data)

    n_ctrl_steps = int(CTRL_TIMESTEP / model.opt.timestep)
    for act_dict in tqdm(actions):  # type: ignore
        if (act_arm := act_dict.get("arm")) and (act_gripper := act_dict.get("gripper")):
            data.ctrl = act_arm + act_gripper
        mj.mj_step(model, data, nstep=n_ctrl_steps)
        renderer.update(data)
        frame = renderer.render()
        recorder.push_frame(frame)

    recorder.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
