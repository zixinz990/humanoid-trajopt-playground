import time
import example_robot_data
import numpy as np
import pinocchio
import crocoddyl
from utils.backflip_util import *


TIMESTEP = 0.01

# Load the simple humanoid robot model
simple_humanoid = example_robot_data.load("simple_humanoid")
q0 = simple_humanoid.model.referenceConfigurations["half_sitting"].copy()
v0 = pinocchio.utils.zero(simple_humanoid.model.nv)
x0 = np.concatenate([q0, v0])

# Base position bounds
pos_lb = -10.0 * np.ones(3)
pos_ub = 10.0 * np.ones(3)

# Base orientation bounds (unit quaternion)
quat_lb = -np.ones(4)
quat_ub = np.ones(4)

# Base linear velocity bounds
lin_vel_lb = -30.0 * np.ones(3)
lin_vel_ub = 30.0 * np.ones(3)

# Base angular velocity bounds
ang_vel_lb = -np.pi * 2 * 20 * np.ones(3)
ang_vel_ub = np.pi * 2 * 20 * np.ones(3)

# Joing position bounds
joint_pos_lb = simple_humanoid.model.lowerPositionLimit[7:]
joint_pos_ub = simple_humanoid.model.upperPositionLimit[7:]

# Joint velocity bounds
joint_vel_limit = np.pi * 2 * 10 * np.ones(simple_humanoid.model.nv - 6)
joint_vel_lb = -joint_vel_limit
joint_vel_ub = joint_vel_limit

# Concatenate all bounds
x_lb = np.concatenate(
    [pos_lb, quat_lb, joint_pos_lb, lin_vel_lb, ang_vel_lb, joint_vel_lb]
)
x_ub = np.concatenate(
    [pos_ub, quat_ub, joint_pos_ub, lin_vel_ub, ang_vel_ub, joint_vel_ub]
)

body = "base_link"
rightFoot = "r_ankle"
leftFoot = "l_ankle"
backflip = BackflipProblem(
    simple_humanoid.model, body, rightFoot, leftFoot, integrator="rk4", control="zero"
)

# An estimated dynamic-feasible backflip
jump_height = 0.5
T = np.sqrt(2 * jump_height / 9.81)
num_flying_knots = int((2 * T / TIMESTEP - 1) / 2)
v_liftoff = np.sqrt(2 * 9.81 * jump_height)

# Set up phases
PHASES = [
    {
        "backflip_stage_1": {
            "jump_height": jump_height,
            "jump_length": [-0.3, 0.0, 0.0],
            "dt": TIMESTEP,
            "num_ground_knots": 30,
            "num_flying_knots": num_flying_knots,
            "v_liftoff": v_liftoff,
            "x_lb": x_lb,
            "x_ub": x_ub,
        }
    },
    {
        "backflip_stage_2": {
            "jump_height": jump_height,
            "jump_length": [-0.5, 0.0, 0.0],
            "dt": TIMESTEP,
            "num_ground_knots": 35,
            "num_flying_knots": num_flying_knots,
            "x_lb": x_lb,
            "x_ub": x_ub,
        }
    },
]

# Solve using FDDP
x_traj = []
solver = [None] * len(PHASES)
for i, phase in enumerate(PHASES):
    for key, value in phase.items():
        if key == "backflip_stage_1":
            solver[i] = crocoddyl.SolverFDDP(
                backflip.create_backflip_problem_first_stage(
                    x0,
                    jump_height=value["jump_height"],
                    jump_length=value["jump_length"],
                    dt=value["dt"],
                    num_ground_knots=value["num_ground_knots"],
                    num_flying_knots=value["num_flying_knots"],
                    v_liftoff=value["v_liftoff"],
                    x_lb=value["x_lb"],
                    x_ub=value["x_ub"],
                )
            )
        if key == "backflip_stage_2":
            solver[i] = crocoddyl.SolverFDDP(
                backflip.create_backflip_problem_second_stage(
                    x0,
                    jump_length=value["jump_length"],
                    dt=value["dt"],
                    num_ground_knots=value["num_ground_knots"],
                    num_flying_knots=value["num_flying_knots"],
                    x_lb=value["x_lb"],
                    x_ub=value["x_ub"],
                )
            )
        solver[i].th_stop = 1e-7
        print("*** SOLVE " + key + " ***")
        solver[i].setCallbacks([crocoddyl.CallbackVerbose(crocoddyl.VerboseLevel(0))])
        xs = [x0] * (solver[i].problem.T + 1)
        us = solver[i].problem.quasiStatic([x0] * solver[i].problem.T)
        solver[i].solve(xs, us, 1000, False)
        x0 = solver[i].xs[-1]
        for x in solver[i].xs:
            x_traj.append(x)

# Display
nq = simple_humanoid.model.nq
display = crocoddyl.MeshcatDisplay(simple_humanoid)
while True:
    for x in x_traj:
        q = x[:nq]
        display.robot.display(q)
        time.sleep(TIMESTEP)
    time.sleep(2.0)
