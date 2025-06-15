import os
import signal
import sys
import time

import example_robot_data
import numpy as np
import pinocchio

import crocoddyl
from utils.jump_util import JumpingProblem

TIMESTEP = 0.01

# Load Talos robot model
talos_full = example_robot_data.load("talos_full")
q0 = talos_full.model.referenceConfigurations["half_sitting"].copy()
v0 = pinocchio.utils.zero(talos_full.model.nv)
x0 = np.concatenate([q0, v0])

body = "torso_2_link"
rf_contact = "right_sole_link"
lf_contact = "left_sole_link"
gait = JumpingProblem(talos_full.model, body, rf_contact, lf_contact, integrator="rk4")

# Set up phases
GAITPHASES = [
    {
        "jumping": {
            "jump_height": 0.2,
            "jump_length": [0.5, 0.5, 0.0],
            "yaw_rotation_deg": -90.0,
            "dt": TIMESTEP,
            "num_ground_knots": 27,
            "num_flying_knots": 33,
        }
    },
    {
        "jumping": {
            "jump_height": 0.2,
            "jump_length": [0.5, -0.5, 0.0],
            "yaw_rotation_deg": 90.0,
            "dt": TIMESTEP,
            "num_ground_knots": 27,
            "num_flying_knots": 33,
        }
    },
]

# Solve using FDDP
x_traj = []
solver = [None] * len(GAITPHASES)
for i, phase in enumerate(GAITPHASES):
    for key, value in phase.items():
        if key == "jumping":
            solver[i] = crocoddyl.SolverFDDP(
                gait.create_jumping_problem(
                    x0,
                    value["jump_height"],
                    value["jump_length"],
                    value["yaw_rotation_deg"],
                    value["dt"],
                    value["num_ground_knots"],
                    value["num_flying_knots"],
                )
            )
        solver[i].th_stop = 1e-7
        print("*** SOLVE " + key + " ***")
        solver[i].setCallbacks([crocoddyl.CallbackVerbose(crocoddyl.VerboseLevel(0))])
        xs = [x0] * (solver[i].problem.T + 1)
        us = solver[i].problem.quasiStatic([x0] * solver[i].problem.T)
        solver[i].solve(xs, us, 200, False)
        x0 = solver[i].xs[-1]
        for x in solver[i].xs:
            x_traj.append(x)

# Display
nq = talos_full.model.nq
display = crocoddyl.MeshcatDisplay(talos_full)
while True:
    for x in x_traj:
        q = x[:nq]
        display.robot.display(q)
        time.sleep(TIMESTEP)
    time.sleep(2.0)
