import os
import signal
import sys
import time

import example_robot_data
import numpy as np
import pinocchio

import crocoddyl
from utils.jump_util import JumpingProblem, plotSolution

WITHDISPLAY = "display" in sys.argv or "CROCODDYL_DISPLAY" in os.environ
WITHPLOT = "plot" in sys.argv or "CROCODDYL_PLOT" in os.environ
signal.signal(signal.SIGINT, signal.SIG_DFL)
TIMESTEP = 0.01

# Load Talos robot model
talos_full = example_robot_data.load("talos_full")
q0 = talos_full.model.referenceConfigurations["half_sitting"].copy()
v0 = pinocchio.utils.zero(talos_full.model.nv)
x0 = np.concatenate([q0, v0])

body = "torso_2_link"
rightFoot = "right_sole_link"
leftFoot = "left_sole_link"
gait = JumpingProblem(talos_full.model, body, rightFoot, leftFoot, integrator="rk4")

# Set up phases
GAITPHASES = [
    {
        "jumping": {
            "jumpHeight": 0.25,
            "jumpLength": [0.5, 0.5, 0.0],
            "yawRotationDeg": 45.0,
            "timeStep": TIMESTEP,
            "groundKnots": 27,
            "flyingKnots": 33,
        }
    },
    {
        "jumping": {
            "jumpHeight": 0.25,
            "jumpLength": [0.5, 0.5, 0.0],
            "yawRotationDeg": 0.0,
            "timeStep": TIMESTEP,
            "groundKnots": 27,
            "flyingKnots": 33,
        }
    }
]

# Solve using FDDP
solver = [None] * len(GAITPHASES)
for i, phase in enumerate(GAITPHASES):
    for key, value in phase.items():
        if key == "jumping":
            solver[i] = crocoddyl.SolverFDDP(
                gait.createJumpingProblem(
                    x0,
                    value["jumpHeight"],
                    value["jumpLength"],
                    value["yawRotationDeg"],
                    value["timeStep"],
                    value["groundKnots"],
                    value["flyingKnots"],
                )
            )
        solver[i].th_stop = 1e-7
        print("*** SOLVE " + key + " ***")
        solver[i].setCallbacks([crocoddyl.CallbackVerbose(crocoddyl.VerboseLevel(0))])
        xs = [x0] * (solver[i].problem.T + 1)
        us = solver[i].problem.quasiStatic([x0] * solver[i].problem.T)
        solver[i].solve(xs, us, 100, False)
        x0 = solver[i].xs[-1]

# Display
nq = talos_full.model.nq
display = crocoddyl.MeshcatDisplay(talos_full)
while True:
    for s in solver:
        for x in s.xs:
            q = x[:nq]
            display.robot.display(q)
            time.sleep(TIMESTEP)
    time.sleep(2.0)
