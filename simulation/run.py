#!/usr/bin/env python
import argparse
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import pybullet as p
import pybullet_data

from squad.constants import GRAVITY


ROOT_DIR = os.path.abspath(os.path.dirname(__file__))

DEFAULT_N_STEPS = 10000
DEFAULT_TIME_STEP = 1.0 / 240.0
DEFAULT_FOOT_FRICTION = 100.0


# Command-line
parser = argparse.ArgumentParser(
    description="Run a PyBullet simulation for a model",
)

# - Arguments
parser.add_argument("name", type=str, help="Name of the URDF file to show")

# - Options
parser.add_argument(
    "-a",
    "--alpha",
    type=float,
    dest="alpha",
    help="Target Alpha angle (if any)",
    default=None,
)
parser.add_argument(
    "-b",
    "--beta",
    type=float,
    dest="beta",
    help="Target Beta angle (if any)",
    default=None,
)
parser.add_argument(
    "-g",
    "--gamma",
    type=float,
    dest="gamma",
    help="Target Gamma angle (if any)",
    default=None,
)

parser.add_argument(
    "-of",
    "--output-foot",
    type=str,
    dest="output_foot",
    default=None,
    help="Foot data to show output for (if any).",
)
parser.add_argument(
    "-f",
    "--free",
    action="store_true",
    help="Fix the body in-place.",
)
parser.add_argument(
    "--ts",
    type=float,
    dest="time_step",
    default=None,
    help="Time step to use (in seconds).",
)
parser.add_argument(
    "-n",
    "--n-steps",
    type=int,
    dest="n_steps",
    default=None,
    help="Number of simulation steps to use.",
)
parser.add_argument(
    "--rt",
    type=bool,
    dest="realtime",
    default=False,
    help="Do a real-time simulation.",
)
parser.add_argument(
    "--gravity",
    type=float,
    default=GRAVITY,
    help="Gravity to use (in m/s^2).",
)
parser.add_argument(
    "--foot-friction",
    type=float,
    dest="foot_friction",
    default=None,
    help="Coefficient of Friction to use for feet.",
)


# Functions


def configure(
    time_step: Optional[float] = None,
    *,
    connect: Optional[Any] = None,
    realtime: bool = False,
    gravity: float = GRAVITY,
) -> None:
    """Configure PyBullet environment."""
    # - Arg handling
    conn_type = connect or p.GUI
    ts_len = time_step or DEFAULT_TIME_STEP

    # - Setup PyBullet
    _ = p.connect(conn_type)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # - Configure physics environment
    p.setGravity(0, 0, gravity)

    p.setRealTimeSimulation(1 if realtime else 0)

    p.setPhysicsEngineParameter(
        fixedTimeStep=ts_len,
        enableFileCaching=0,
        enableConeFriction=0,
        erp=1e-1,
        contactERP=0.01,
        frictionERP=0.01,
    )

    # - Setup space
    p.loadURDF("plane.urdf")


def setup(
    urdf_file: str,
    start_pos: Optional[List[float]] = None,
    *,
    foot_friction: Optional[float] = None,
    fixed: bool = True,
) -> Tuple[int, Dict[str, int]]:
    """Sets up the URDF body in the PyBullet simulation."""
    foot_friction = (
        foot_friction if foot_friction is not None else DEFAULT_FOOT_FRICTION
    )
    if not start_pos:
        if fixed:
            start_pos = [0, 0, 0.5]
        else:
            start_pos = [0, 0, 0.15]

    # - Load body
    obj_path = os.path.join(ROOT_DIR, "urdf", urdf_file)
    obj_id = p.loadURDF(obj_path, start_pos, useFixedBase=fixed)

    # - Get joint data
    joint_ids = {}
    for i in range(p.getNumJoints(obj_id)):
        j_info = p.getJointInfo(obj_id, i)
        j_name = j_info[1].decode("utf-8")
        j_type = j_name.split("_")[-1]
        if j_type in ("foot", "knee", "shoulder"):
            p.changeDynamics(
                obj_id,
                i,
                lateralFriction=foot_friction,
            )
        else:
            # - Servo joint
            p.changeDynamics(
                obj_id,
                i,
                lateralFriction=1e-5,
                linearDamping=0.0,
                angularDamping=0.0,
                maxJointVelocity=3.703,
            )
            joint_ids[j_name] = j_info[0]

    return obj_id, joint_ids


def move_joint(obj_id: int, joint_id: int, angle: float) -> None:
    """Moves a single specified joint to the target angle."""
    p.setJointMotorControl2(
        obj_id,
        joint_id,
        p.POSITION_CONTROL,
        targetPosition=angle,
        force=2.0,
    )


def init_step_data(
    t1: Optional[float] = None,
    t2: Optional[float] = None,
    t3: Optional[float] = None,
) -> Dict[str, Any]:
    """Initializes the data dictionary to use for step updates."""
    data: Dict[str, Any] = dict(
        min_max={
            "hip": (math.radians(-45.0), math.radians(45.0)),
            "femur": (math.radians(-90.0), math.radians(90.0)),
            "leg": (math.radians(-55.0), math.radians(33.0)),
        },
        ang_dirs={
            "hip": 1.0,
            "femur": 1.0,
            "leg": 1.0,
        },
        ang_pos={
            "fl": {"hip": 1.0, "femur": 1.0, "leg": 1.0},
            "fr": {"hip": -1.0, "femur": 1.0, "leg": 1.0},
            "bl": {"hip": 1.0, "femur": 1.0, "leg": 1.0},
            "br": {"hip": -1.0, "femur": 1.0, "leg": 1.0},
        },
    )
    data["ang_incs"] = {
        k: (1.0 / n_frames) * 2 * (v[1] - v[0])
        for k, v in data["min_max"].items()
    }
    data["ang_tgts"] = {
        "hip": math.radians(t1) if t1 is not None else None,
        "femur": math.radians(t2) if t2 is not None else None,
        "leg": math.radians(t3) if t3 is not None else None,
    }
    data["angles"] = {
        k: {
            "hip": 0.0,
            "femur": 0.0,
            "leg": 0.0,
        }
        for k in data["ang_pos"]
    }
    return data


def step_update(
    obj_id: int,
    joint_ids: Dict[str, int],
    data: Dict[str, Any],
) -> Dict[str, Any]:
    """Update function for each simulation step."""
    for j_n, j_i in joint_ids.items():
        # - Get leg data
        j_s = j_n.split("_")
        j_t = j_s[-1]
        j_p = j_s[0]

        # - Update angles
        t_ang = data["angles"][j_p][j_t]
        t_tang = data["ang_tgts"].get(j_t)
        t_inc = data["ang_incs"][j_t] * (
            data["ang_dirs"][j_t] * data["ang_pos"][j_p][j_t]
        )

        if t_tang is not None:
            t_tang *= data["ang_pos"][j_p][j_t]

            t_d = t_tang - t_ang
            t_dabs = abs(t_d)
            if t_dabs > 0.0:
                if t_dabs < abs(t_inc):
                    t_inc = t_d
                else:
                    if t_d > 0.0:
                        t_inc = abs(t_inc)
                    else:
                        t_inc = -abs(t_inc)
            else:
                continue

        t_ang += t_inc
        if not (data["min_max"][j_t][0] <= t_ang <= data["min_max"][j_t][1]):
            data["ang_dirs"][j_t] *= -1
        data["angles"][j_p][j_t] = t_ang

        # - Set new joint angle
        move_joint(obj_id, j_i, t_ang)

    return data


# Simulation

if __name__ == "__main__":
    # - Command-line args
    args = parser.parse_args()

    fname = args.name
    if "." not in fname:
        fname += ".urdf"

    n_frames = args.n_steps or DEFAULT_N_STEPS

    # - Setup PyBullet
    configure(args.time_step, realtime=args.realtime, gravity=args.gravity)

    # - Load body
    body_id, joint_ids = setup(fname, fixed=not args.free)

    # - Get step data
    data = init_step_data(args.alpha, args.beta, args.gamma)

    # - Run simulation loop
    def _while_cond_ns(_: int) -> bool:
        return True

    def _while_cond_ws(x: int) -> bool:
        return x < n_frames

    if args.n_steps is None:
        while_cond = _while_cond_ns
    else:
        while_cond = _while_cond_ws

    i = 0
    while while_cond(i):
        step_update(body_id, joint_ids, data)
        p.stepSimulation()
        if not args.realtime:
            time.sleep(DEFAULT_TIME_STEP)
        i += 1

    # - Output
    output_foot = args.output_foot

    h_pos = None
    f_pos = None
    if output_foot:
        output_foot = output_foot.strip().lower()

        a_str = ", ".join(
            f"{k} = {math.degrees(v):.2f}"
            for k, v in data["angles"][output_foot].items()
        )
        print("\n***")
        print(f"*** ANGLES: {a_str}")

        h_state = None
        f_state = None
        for i in range(p.getNumJoints(body_id)):
            j_info = p.getJointInfo(body_id, i)
            j_name = j_info[1].decode("utf-8")
            if j_name.startswith(output_foot):
                if j_name.endswith("foot"):
                    f_state = p.getLinkState(body_id, i)
                elif j_name.endswith("hip"):
                    h_state = p.getLinkState(body_id, i)
            if h_state is not None and f_state is not None:
                break

        b_pos, b_orn = p.getBasePositionAndOrientation(body_id)
        if h_state is not None and f_state is not None:
            h_pos = h_state[0]
            f_pos = f_state[0]

            b_pos_s = tuple(f"{x:>8}" for x in (f"{y:.5f}" for y in b_pos))
            h_pos_s = tuple(f"{x:>8}" for x in (f"{y:.5f}" for y in h_pos))
            f_pos_s = tuple(f"{x:>8}" for x in (f"{y:.5f}" for y in f_pos))

            print("***")
            print(f"*** BODY = ({', '.join(b_pos_s)})")
            print(f"*** HIP  = ({', '.join(h_pos_s)})")
            print(f"*** FOOT = ({', '.join(f_pos_s)})")

        print("***\n")

    p.disconnect()
    print()

    if h_pos is not None and f_pos is not None:
        f_pos_s = [f"{x:.4f}" for x in f_pos]
        f_pos_s = [f"{x:>8}" for x in f_pos_s]
        print(f"[FINAL]   {output_foot.upper()} = ({', '.join(f_pos_s)})")
        print()
