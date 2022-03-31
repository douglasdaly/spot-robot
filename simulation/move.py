#!/usr/bin/env python
import argparse
from datetime import datetime
import math
import os
import signal
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pybullet as p
import pybullet_data

from squad import motion as sqm
from squad.constants import GRAVITY, Leg, TimeType
from squad.kinematics import KinematicSolver
from squad.motion.states import RobotState


ROOT_DIR = os.path.abspath(os.path.dirname(__file__))

NEUTRAL_ANGLES = (0.0, -35.0, 10.0)

DEFAULT_N_STEPS = 10000
DEFAULT_TIME_STEP = 1.0 / 240.0
DEFAULT_FRICTION = 1e-5
DEFAULT_FOOT_FRICTION = 1e-6
DEFAULT_SERVO_TORQUE = 5.0

KEEP_GOING = True

BODY_ID: int = 0
JOINT_IDS: Dict[str, int] = {}
FOOT_IDS: Dict[str, int] = {}


# Command-line
parser = argparse.ArgumentParser(
    description="Run a PyBullet simulation for a model",
)

# - Arguments
parser.add_argument(
    "movement",
    type=str,
    help="Names of the movement to execute",
)

# - Options
parser.add_argument(
    "-r",
    "--repeat",
    dest="repeat",
    action="store_true",
    help="Whether or not to repeat the movement after finishing.",
)
parser.add_argument(
    "-l",
    "--loop",
    dest="loop",
    action="store_true",
    help="Whether or not the movement should be set to loop.",
)
parser.add_argument(
    "-en",
    "--end-neutral",
    dest="end_neutral",
    action="store_true",
    help="Whether or not to go back to neutral stance after moving.",
)
parser.add_argument(
    "-i",
    "--input-file",
    type=str,
    dest="file",
    default="robot-simple.urdf",
    help="Name of the URDF file to use",
)
parser.add_argument(
    "-o",
    "--output",
    type=str,
    dest="output",
    default=None,
    help="Output file to write step data to (if any).",
)
parser.add_argument(
    "--output-every",
    type=int,
    dest="output_res",
    default=10,
    help="Output every X steps to the output file (if any).",
)
parser.add_argument(
    "-w",
    "--warm-up",
    type=float,
    dest="warm_up",
    default=2.5,
    help="Wait time to allow graphics to initialize/warm-up.",
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
    "--print-interval",
    type=int,
    dest="print_interval",
    default=5,
    help="Interval between printing status updates (in seconds).",
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
    "-rt",
    "--real-time",
    dest="realtime",
    action="store_true",
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


# Control functions


def sigint_handler(sig, frame) -> None:
    """SIGINT handler function."""
    global KEEP_GOING

    print("\nShutting down...\n", flush=True)
    KEEP_GOING = False


# Simulation Functions


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

    if not realtime:
        p.setPhysicsEngineParameter(
            fixedTimeStep=ts_len,
            enableFileCaching=0,
            enableConeFriction=0,
            erp=1e-1,
            contactERP=0.01,
            frictionERP=0.01,
        )
    else:
        p.setPhysicsEngineParameter(
            enableFileCaching=0,
            enableConeFriction=0,
            erp=1e-1,
            contactERP=0.01,
            frictionERP=0.01,
        )

    # - Load plane/base
    plane_id = p.loadURDF("plane.urdf")
    p.changeDynamics(
        plane_id,
        -1,
        lateralFriction=DEFAULT_FRICTION,
    )


def setup(
    urdf_file: str,
    initial_state: RobotState,
    start_pos: Optional[List[float]] = None,
    *,
    foot_friction: Optional[float] = None,
    fixed: bool = True,
) -> Tuple[int, Dict[str, int], Dict[str, int]]:
    """Sets up the URDF body in the PyBullet simulation."""
    foot_friction = (
        foot_friction if foot_friction is not None else DEFAULT_FOOT_FRICTION
    )

    # - Load body
    obj_path = os.path.join(ROOT_DIR, "urdf", urdf_file)
    obj_id = p.loadURDF(obj_path, [0.0, 0.0, 0.5], useFixedBase=fixed)

    # - Get joint data
    joint_ids = {}
    foot_ids = {}
    for i in range(p.getNumJoints(obj_id)):
        j_info = p.getJointInfo(obj_id, i)
        j_name = j_info[1].decode("utf-8")
        j_type = j_name.split("_")[-1]
        if j_type in ("foot", "knee", "shoulder"):
            if j_type == "foot":
                foot_ids[j_name[:2].upper()] = j_info[0]
                p.changeDynamics(
                    obj_id,
                    i,
                    linearDamping=0,
                    angularDamping=0,
                    lateralFriction=foot_friction,
                    restitution=0,
                )
            else:
                p.changeDynamics(
                    obj_id,
                    i,
                    lateralFriction=DEFAULT_FRICTION,
                )
        else:
            # - Servo joint
            p.changeDynamics(
                obj_id,
                i,
                lateralFriction=foot_friction,
                linearDamping=0.0,
                angularDamping=0.0,
                maxJointVelocity=3.703,
            )
            joint_ids[j_name] = j_info[0]

            # - Set initial angles
            leg = getattr(Leg, j_name[:2].upper())
            ang_nm = f"{j_type}_theta"
            i_ang = getattr(initial_state.legs[leg], ang_nm)
            if j_type == "hip" and leg % 2 == 0:
                i_ang *= -1.0
            elif j_type == "leg":
                i_ang *= -1.0
            move_joint(obj_id, j_info[0], i_ang, reset=True)

    # - Set initial position
    if start_pos:
        _, curr_orn = p.getBasePositionAndOrientation(obj_id)
        p.resetBasePositionAndOrientation(obj_id, start_pos, curr_orn)

    return obj_id, joint_ids, foot_ids


def move_joint(
    obj_id: int,
    joint_id: int,
    angle: float,
    *,
    reset: bool = False,
) -> None:
    """Moves a single specified joint to the target angle."""
    t_value = math.radians(angle)
    if reset:
        p.resetJointState(
            obj_id,
            joint_id,
            t_value,
        )
    else:
        p.setJointMotorControl2(
            obj_id,
            joint_id,
            p.POSITION_CONTROL,
            targetPosition=math.radians(angle),
            force=DEFAULT_SERVO_TORQUE,
        )
    return


def update_env(
    state: sqm.states.RobotState,
    solver: KinematicSolver,
    *,
    reset: bool = False,
) -> None:
    """Updates the PyBullet environment/robot to the given state."""
    for leg in state.legs:
        j_pre = f"{leg.leg.name.lower()}_j_"

        if reset:
            t_h = leg.hip_theta
            t_f = leg.femur_theta
            t_l = leg.leg_theta
        else:
            t_h, t_f, t_l = solver.foot_inv(
                leg.leg,
                leg.x,
                leg.y,
                leg.z,
                state.roll,
                state.pitch,
                state.yaw,
                knee_angle=True,
            )

        if leg.leg % 2 == 0:
            t_h *= -1.0
        t_l *= -1.0

        move_joint(BODY_ID, JOINT_IDS[f"{j_pre}body_hip"], t_h, reset=reset)
        move_joint(BODY_ID, JOINT_IDS[f"{j_pre}hip_femur"], t_f, reset=reset)
        move_joint(BODY_ID, JOINT_IDS[f"{j_pre}femur_leg"], t_l, reset=reset)

    return


def update_state(state: sqm.states.RobotState) -> None:
    """Updates the RPY of the robot's state from the environment."""
    curr_pos, curr_orn = p.getBasePositionAndOrientation(BODY_ID)
    curr_rpy = p.getEulerFromQuaternion(curr_orn)

    p_x, p_y, p_z = curr_pos
    o_r, o_p, o_y = curr_rpy

    state._x = p_x * 1000.0
    state._y = p_y * 1000.0
    state._z = p_z * 1000.0
    state._roll = math.degrees(o_r)
    state._pitch = -math.degrees(o_p)
    state._yaw = math.degrees(o_y)


# - Helper functions


def print_state(
    state: sqm.states.RobotState,
    title: Optional[str] = None,
    timestamp: Optional[float] = None,
    include_env: bool = False,
) -> None:
    """Output the robot state to the console."""
    leg_lines = [
        f"- {x.leg.name.upper()} = {x.hip_theta:.2f}, {x.femur_theta:.2f},"
        f" {x.leg_theta:.2f} ({x.x:.1f}, {x.y:.1f}, {x.z:.1f})"
        for x in state.legs
    ]
    if title:
        p_title = title
    else:
        p_title = "RobotState"

    if timestamp:
        ts_str = datetime.fromtimestamp(timestamp).strftime("%H:%M:%S.%f")
        p_title += f" @ {ts_str}"

    if include_env:
        eleg_lines = []
        for x in state.legs:
            t_fid = FOOT_IDS[x.leg.name.upper()]
            t_f_state = p.getLinkState(BODY_ID, t_fid)
            t_pos = t_f_state[4]
            eleg_lines.append(
                f"- {x.leg.name.upper()} = ({1000.0 * t_pos[0]:.1f},"
                f" {1000.0 * t_pos[1]:.1f}, {1000.0 * t_pos[2]:.1f})"
            )
    else:
        eleg_lines = None

    print(p_title)
    print("=" * len(p_title))
    print(f"Pos: ({state.x:.1f}, {state.y:.1f}, {state.z:.1f})")
    print(f"Orn: ({state.roll:.1f}, {state.pitch:.1f}, {state.yaw:.1f})")
    for ll in leg_lines:
        print(ll)
    if eleg_lines:
        print("PyBullet Feet:")
        for el in eleg_lines:
            print(el)
    print()


def print_movements(
    movements: List[sqm.Movement],
    title: Optional[str] = None,
    timestamp: Optional[float] = None,
) -> None:
    """Output the movement state(s) to the console."""
    mvmt_lines = [
        f"- {x.name} = {x._controller.progress:.1%}" for x in movements
    ]
    if title:
        p_title = title
    else:
        p_title = "Movements"

    if timestamp:
        ts_str = datetime.fromtimestamp(timestamp).strftime("%H:%M:%S.%f")
        p_title += f" @ {ts_str}"

    print(p_title)
    print("-" * len(p_title))
    for ml in mvmt_lines:
        print(ml)
    print()


def prep_output_file(file: str, x_name: str, y_names: Iterable[str]) -> str:
    """Prepares the output CSV file to write simulation data to."""
    if not file.lower().endswith(".csv"):
        f_path = f"{file}.csv"
    else:
        f_path = file

    if os.path.exists(f_path):
        os.remove(f_path)

    items = [x_name]
    items.extend(y_names)
    hdr_line = ",".join(items) + "\n"

    with open(f_path, "w") as fout:
        fout.write(hdr_line)

    return f_path


def write_output_line(file: str, x: float, ys: Iterable[Any]) -> None:
    """Writes a single line of results/output to the file."""
    items = [datetime.fromtimestamp(x).strftime("%Y-%m-%d %H:%M:%S.%f")]
    for y in ys:
        items.append(f"{y}")
    new_line = ",".join(items) + "\n"
    with open(file, "a") as fout:
        fout.write(new_line)
    return


# Movements


def get_movements(
    name: str,
    state: sqm.states.RobotState,
    *,
    loop: bool = False,
    repeat: bool = False,
) -> List[sqm.Movement]:
    """Gets the movement to execute with the given `name`."""
    if name == "neutral":
        angs = NEUTRAL_ANGLES
        m_ctrls = [
            sqm.controllers.LegLinearThetas(state.legs.fl, *angs),
            sqm.controllers.LegLinearThetas(state.legs.fr, *angs),
            sqm.controllers.LegLinearThetas(state.legs.bl, *angs),
            sqm.controllers.LegLinearThetas(state.legs.br, *angs),
        ]
        return [
            sqm.Movement(
                f"{x.state.leg.name.lower()}_m_{name}",
                x,
                1.0,
                loop=loop,
                repeat=repeat,
            )
            for x in m_ctrls
        ]
    elif name == "squat":
        # - Squat stance
        angs = (25.0, -40.0, 10.0)
        m_ctrls = [
            sqm.controllers.LegLinearThetas(state.legs.fl, *angs),
            sqm.controllers.LegLinearThetas(state.legs.fr, *angs),
            sqm.controllers.LegLinearThetas(state.legs.bl, *angs),
            sqm.controllers.LegLinearThetas(state.legs.br, *angs),
        ]
        return [
            sqm.Movement(
                f"{x.state.leg.name.lower()}_m_{name}",
                x,
                6.0,
                loop=loop,
                repeat=repeat,
            )
            for x in m_ctrls
        ]
    elif name == "test":
        # - Test stance
        l_pts = [
            (35.0, 30.0, 30.0),
            (35.0, 30.0, 25.0),
        ]
        r_pts = [
            (35.0, -30.0, 30.0),
            (35.0, -30.0, 25.0),
        ]
        m_ctrls = [
            sqm.controllers.LegBezierDeltas(state.legs.fl, l_pts),
            sqm.controllers.LegBezierDeltas(state.legs.fr, r_pts),
            sqm.controllers.LegBezierDeltas(state.legs.bl, l_pts),
            sqm.controllers.LegBezierDeltas(state.legs.br, r_pts),
        ]
        return [
            sqm.Movement(
                f"{x.state.leg.name.lower()}_m_{name}",
                x,
                2.5,
                TimeType.SECOND,
                loop=loop,
                repeat=repeat,
            )
            for x in m_ctrls
        ]
    elif name == "fallfwd":
        # - Fall forward
        l_angs = (0.0, 0.0, -10.0)
        m_ctrls = [
            sqm.controllers.leg_linear.LegLinearThetas(state.legs.fl, *l_angs),
            sqm.controllers.leg_linear.LegLinearThetas(state.legs.fr, *l_angs),
            sqm.controllers.leg_linear.LegLinearThetas(state.legs.bl, *l_angs),
            sqm.controllers.leg_linear.LegLinearThetas(state.legs.br, *l_angs),
        ]
        return [
            sqm.Movement(
                f"{x.state.leg.name.lower()}_m_{name}",
                x,
                2.0,
                TimeType.SECOND,
                loop=loop,
                repeat=repeat,
            )
            for x in m_ctrls
        ]
    elif name == "step":
        # - Single step
        d_x = 20.0
        d_y = 25.0
        d_z = 35.0
        fl_pts = [
            (d_x, d_y, d_z),
            (d_x, d_y, d_z),
            (d_x, -d_y, -d_z),
            (d_x, -d_y, -d_z),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        ]
        fr_pts = [
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (d_x, -d_y, d_z),
            (d_x, -d_y, d_z),
            (d_x, d_y, -d_z),
            (d_x, d_y, -d_z),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        ]
        bl_pts = [
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (d_x, d_y, d_z),
            (d_x, d_y, d_z),
            (d_x, -d_y, -d_z),
            (d_x, -d_y, -d_z),
        ]
        br_pts = [
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (d_x, -d_y, d_z),
            (d_x, -d_y, d_z),
            (d_x, d_y, -d_z),
            (d_x, d_y, -d_z),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        ]
        m_ctrls = [
            sqm.controllers.LegBezierDeltas(state.legs.fl, fl_pts),
            sqm.controllers.LegBezierDeltas(state.legs.fr, fr_pts),
            sqm.controllers.LegBezierDeltas(state.legs.bl, bl_pts),
            sqm.controllers.LegBezierDeltas(state.legs.br, br_pts),
        ]
        return [
            sqm.Movement(
                f"{x.state.leg.name.lower()}_m_{name}",
                x,
                5.0,
                loop=loop,
                repeat=repeat,
            )
            for x in m_ctrls
        ]

    raise NotImplementedError(name)


# Simulation

if __name__ == "__main__":
    # - Command-line args
    args = parser.parse_args()

    mvmt_name = args.movement.strip().lower()

    fname = args.file
    if "." not in fname:
        fname += ".urdf"

    if args.realtime:
        do_rt = True
    else:
        do_rt = False

    if args.loop:
        do_loop = True
    else:
        do_loop = False

    if args.repeat:
        do_repeat = True
    else:
        do_repeat = False

    if args.free:
        do_fixed = False
    else:
        do_fixed = True

    if args.end_neutral:
        go_back_to_neutral = True
    else:
        go_back_to_neutral = False

    # - Create robot state object & solver
    init_angs = NEUTRAL_ANGLES
    robot_state = sqm.states.RobotState(
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        [
            sqm.states.LegState.from_thetas(Leg.FL, *init_angs),
            sqm.states.LegState.from_thetas(Leg.FR, *init_angs),
            sqm.states.LegState.from_thetas(Leg.BL, *init_angs),
            sqm.states.LegState.from_thetas(Leg.BR, *init_angs),
        ],
    )
    robot_kms = KinematicSolver(robot_state.body)

    if not do_fixed:
        init_height = -min(x.z for x in robot_state.legs)
        init_height -= 75.0 / 2.0
        init_height /= 1000.0
    else:
        init_height = 0.5

    # - Setup PyBullet & load body
    configure(args.time_step, realtime=do_rt, gravity=args.gravity)

    BODY_ID, JOINT_IDS, FOOT_IDS = setup(
        fname,
        robot_state,
        start_pos=[0.0, 0.0, init_height],
        fixed=do_fixed,
    )

    # - Get specified movement
    movements = get_movements(
        mvmt_name,
        robot_state,
        loop=do_loop,
        repeat=do_repeat,
    )

    # - Wait for graphics to load/warm-up
    print()
    print("***** INITIAL *****")
    print_state(robot_state)
    print()

    if args.output:
        cols = []
        for m in movements:
            cols.append(m.name)
        for v in ("fl", "fr", "bl", "br"):
            cols.extend(
                (
                    f"{v}_x",
                    f"{v}_y",
                    f"{v}_z",
                    f"{v}_hip",
                    f"{v}_femur",
                    f"{v}_leg",
                )
            )
        out_path = prep_output_file(args.output, "timestamp", cols)
    else:
        out_path = None

    if args.warm_up:
        time.sleep(args.warm_up)

    # - If not fixed start simulation to neutral
    if not do_fixed:
        neutral_mvmts = get_movements("neutral", robot_state)
        for n_mvmt in neutral_mvmts:
            n_mvmt.start()

        at_neutral = False
        while not at_neutral:
            c_ts = time.time()
            for n_mvmt in neutral_mvmts:
                n_mvmt.update(c_ts)

            update_env(robot_state, robot_kms)
            p.stepSimulation()
            update_state(robot_state)

            if not do_rt:
                time.sleep(DEFAULT_TIME_STEP)

            at_neutral = all(x.progress >= 1.0 for x in neutral_mvmts)

    # - Add SIGINT handler
    signal.signal(signal.SIGINT, sigint_handler)

    # - Prepare main simulation loop
    def _while_cond_ns(_: int) -> bool:
        return KEEP_GOING

    def _while_cond_ws(x: int) -> bool:
        if KEEP_GOING:
            return x < args.n_steps
        return False

    if args.n_steps is None:
        while_cond = _while_cond_ns
    else:
        while_cond = _while_cond_ws

    m_ts = time.time()
    for mvmt in movements:
        mvmt.start(m_ts)

    # - Simulation loop
    i = 0
    last_print = time.time()
    last_write = time.time()
    while while_cond(i):
        m_ts = time.time()
        for mvmt in movements:
            mvmt.update(m_ts)

        update_env(robot_state, robot_kms)
        p.stepSimulation()
        update_state(robot_state)

        if out_path:
            if do_rt:
                d_write = time.time() - last_write
                m_write = d_write >= args.output_res
            else:
                m_write = i % args.output_res == 0

            if m_write:
                write_output_line(
                    out_path,
                    m_ts,
                    tuple(m._controller.progress for m in movements)
                    + (
                        # - FL
                        robot_state.legs.fl.x,
                        robot_state.legs.fl.y,
                        robot_state.legs.fl.z,
                        robot_state.legs.fl.hip_theta,
                        robot_state.legs.fl.femur_theta,
                        robot_state.legs.fl.leg_theta,
                        # - FR
                        robot_state.legs.fr.x,
                        robot_state.legs.fr.y,
                        robot_state.legs.fr.z,
                        robot_state.legs.fr.hip_theta,
                        robot_state.legs.fr.femur_theta,
                        robot_state.legs.fr.leg_theta,
                        # - BL
                        robot_state.legs.bl.x,
                        robot_state.legs.bl.y,
                        robot_state.legs.bl.z,
                        robot_state.legs.bl.hip_theta,
                        robot_state.legs.bl.femur_theta,
                        robot_state.legs.bl.leg_theta,
                        # - BR
                        robot_state.legs.br.x,
                        robot_state.legs.br.y,
                        robot_state.legs.br.z,
                        robot_state.legs.br.hip_theta,
                        robot_state.legs.br.femur_theta,
                        robot_state.legs.br.leg_theta,
                    ),
                )
                if do_rt:
                    last_write = time.time()

        if do_rt:
            p_delta = time.time() - last_print
            if p_delta >= args.print_interval:
                print_state(robot_state, timestamp=m_ts, include_env=True)
                print_movements(movements)
                print()
                last_print = time.time()
        else:
            if i % (round(1.0 / DEFAULT_TIME_STEP) * args.print_interval) == 0:
                print_state(robot_state, timestamp=m_ts, include_env=True)
                print_movements(movements)
                print()

            time.sleep(DEFAULT_TIME_STEP)

        if not (do_loop or do_repeat) and all(
            x.progress >= 1.0 for x in movements
        ):
            break

        i += 1

    # - Back to neutral
    if go_back_to_neutral:
        neutral_mvmts = get_movements("neutral", robot_state)
        for n_mvmt in neutral_mvmts:
            n_mvmt.start()

        at_neutral = False
        while not at_neutral:
            c_ts = time.time()
            for n_mvmt in neutral_mvmts:
                n_mvmt.update(c_ts)

            update_env(robot_state, robot_kms)
            p.stepSimulation()
            update_state(robot_state)

            if not do_rt:
                time.sleep(DEFAULT_TIME_STEP)

            at_neutral = all(x.progress >= 1.0 for x in neutral_mvmts)

    print_state(robot_state, timestamp=m_ts, include_env=True)

    time.sleep(5.0)

    p.disconnect()
