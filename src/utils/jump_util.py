import numpy as np
import pinocchio

import crocoddyl

from typing import List


class JumpingProblem:
    """Build simple bipedal locomotion problems.

    This class aims to build simple locomotion problems used in the examples of
    Crocoddyl.
    The scope of this class is purely for academic reasons, and it does not aim to be
    used in any robotics application.
    We also do not consider it as part of the API, so changes in this class will not
    pass through a strict process of deprecation.
    Thus, we advice any user to DO NOT develop their application based on this class.
    """

    def __init__(
        self,
        rmodel,
        body,
        rightFoot,
        leftFoot,
        integrator="euler",
        control="zero",
        fwddyn=True,
    ):
        """Construct biped-gait problem.

        :param rmodel:     robot model
        :param rightFoot:  name of the right foot
        :param leftFoot:   name of the left foot
        :param integrator: type of the integrator
                           (options are: 'euler', and 'rk4')
        :param control:    type of control parametrization
                           (options are: 'zero', 'one', and 'rk4')
        """
        self.rmodel = rmodel
        self.rdata = rmodel.createData()
        self.state = crocoddyl.StateMultibody(self.rmodel)
        self.actuation = crocoddyl.ActuationModelFloatingBase(self.state)
        self._integrator = integrator
        self._control = control
        self._fwddyn = fwddyn

        # Getting the frame id for body and legs
        self.bodyId = self.rmodel.getFrameId(body)
        self.rfId = self.rmodel.getFrameId(rightFoot)
        self.lfId = self.rmodel.getFrameId(leftFoot)

        # Defining default state
        q0 = self.rmodel.referenceConfigurations["half_sitting"]
        self.rmodel.defaultState = np.concatenate([q0, np.zeros(self.rmodel.nv)])
        self.firstStep = True

        # Defining the friction coefficient and normal
        self.mu = 0.8
        self.Rsurf = np.eye(3)

    def createJumpingProblem(
        self,
        x0: np.ndarray,
        jumpHeight: float,
        jumpLength: List[float],
        jumpYawRotDeg: float,
        timeStep: float,
        numGroundKnots: int,
        numFlyingKnots: int,
    ):
        """
        Create a shooting problem for jumping.

        :param x0:            initial state
        :param jumpHeight:    height of body at the peak
        :param jumpLength:    length of jumping
        :param jumpYawRotDeg: rotated Yaw angle of the body during jumping (degree)
        :param timeStep:      time step length
        :numGroundKnots:      number of knots during th etake-off and landing phase
        :numFlyingKnots:      number of knots during the flying-up and flying-down phase
        """
        ######################################################################################################
        jumpYawRot = np.deg2rad(jumpYawRotDeg)
        loco3dModel = []

        # Get initial feet position (global frame)
        q0 = x0[: self.rmodel.nq]
        pinocchio.forwardKinematics(self.rmodel, self.rdata, q0)
        pinocchio.updateFramePlacements(self.rmodel, self.rdata)
        rfPos0 = self.rdata.oMf[self.rfId].translation
        lfPos0 = self.rdata.oMf[self.lfId].translation

        # Get initial body position (global frame)
        bodyPos0 = self.rdata.oMf[self.bodyId].translation

        # Set initial body position ref (global frame)
        bodyPosRef0 = (rfPos0 + lfPos0) / 2
        bodyPosRef0[2] = bodyPos0[2]

        # TAKE-OFF PHASE #####################################################################################
        takeOff = [
            self.createKnotActionModel(
                timeStep,
                [self.lfId, self.rfId],
            )
            for _ in range(numGroundKnots)
        ]

        # FLYING-UP PHASE ####################################################################################
        flyingUpPhase = []
        lastBodyPosRef = np.zeros(3)
        lastBodyYawRef = 0.0
        for k in range(numFlyingKnots):
            # Body position reference
            bodyPosRef = (
                np.array(
                    [
                        jumpLength[0] / 2.0,
                        jumpLength[1] / 2.0,
                        jumpLength[2] / 2.0 + jumpHeight,
                    ]
                )
                * (k + 1)
                / numFlyingKnots
                + bodyPosRef0
            )
            lastBodyPosRef = bodyPosRef

            # Body Yaw reference
            bodyYawRef = jumpYawRot / 2.0 * (k + 1) / numFlyingKnots
            lastBodyYawRef = bodyYawRef

            # Body SE3 reference
            bodyPoseRef = pinocchio.SE3(
                pinocchio.rpy.rpyToMatrix(0.0, 0.0, bodyYawRef), bodyPosRef
            )

            # Create the action model and append
            flyingUpPhase.append(
                self.createKnotActionModel(timeStep, [], bodyPoseRef=bodyPoseRef)
            )

        # FLYING-DOWN PHASE ##################################################################################
        flyingDownPhase = []
        for _ in range(numFlyingKnots):
            # Body position reference
            bodyPosRef = (
                np.array(
                    [
                        jumpLength[0] / 2.0,
                        jumpLength[1] / 2.0,
                        jumpLength[2] / 2.0 - jumpHeight,
                    ]
                )
                * (k + 1)
                / numFlyingKnots
                + lastBodyPosRef
            )

            # Body Yaw reference
            bodyYawRef = lastBodyYawRef + jumpYawRot / 2.0 * (k + 1) / numFlyingKnots

            # Body SE3 reference
            bodyPoseRef = pinocchio.SE3(
                pinocchio.rpy.rpyToMatrix(0.0, 0.0, bodyYawRef), bodyPosRef
            )

            # Create the action model and append
            flyingUpPhase.append(
                self.createKnotActionModel(timeStep, [], bodyPoseRef=bodyPoseRef)
            )

        # LANDING PHASE ######################################################################################
        RotRefFinal = pinocchio.rpy.rpyToMatrix(0.0, 0.0, jumpYawRot)
        feetPoseRefFinal = [
            [self.lfId, pinocchio.SE3(RotRefFinal, RotRefFinal @ lfPos0 + jumpLength)],
            [self.rfId, pinocchio.SE3(RotRefFinal, RotRefFinal @ rfPos0 + jumpLength)],
        ]
        landingPhase = [
            self.createFootSwitchModel([self.lfId, self.rfId], feetPoseRefFinal, False)
        ]
        landed = [
            self.createKnotActionModel(
                timeStep,
                [self.lfId, self.rfId],
                bodyPoseRef=pinocchio.SE3(RotRefFinal, bodyPos0 + jumpLength),
            )
            for _ in range(int(numGroundKnots / 2))
        ]

        loco3dModel += takeOff
        loco3dModel += flyingUpPhase
        loco3dModel += flyingDownPhase
        loco3dModel += landingPhase
        loco3dModel += landed

        return crocoddyl.ShootingProblem(x0, loco3dModel[:-1], loco3dModel[-1])

    def createKnotActionModel(
        self,
        timeStep: float,
        supportFootIds: List[int],
        comPosRef: np.ndarray = None,
        bodyPoseRef: pinocchio.SE3 = None,
        swingFootPoseRef: pinocchio.SE3 = None,
    ):
        """
        Create an action model for one knot.

        :param timeStep:         time step length in second
        :param supportFootIds:   IDs of support feet
        :param comPosRef:        CoM position reference
        :param bodyPoseRef:      body SE3 reference
        :param swingFootPoseRef: swing feet SE3 reference
        """
        # Create a 6D multi-contact model for support feet
        if self._fwddyn:
            nu = self.actuation.nu
        else:
            nu = self.state.nv + 6 * len(supportFootIds)
        contactModel = crocoddyl.ContactModelMultiple(self.state, nu)  # a container
        for footId in supportFootIds:
            supportContactModel = crocoddyl.ContactModel6D(
                self.state,
                footId,
                pinocchio.SE3.Identity(),
                pinocchio.LOCAL_WORLD_ALIGNED,
                nu,
                np.array([0.0, 30.0]),
            )
            contactModel.addContact(
                self.rmodel.frames[footId].name + "_contact", supportContactModel
            )

        # Initialize cost-sum model
        costModel = crocoddyl.CostModelSum(self.state, nu)

        # Cost of tracking CoM position reference (if needed)
        if comPosRef is not None:
            comPosResidual = crocoddyl.ResidualModelCoMPosition(
                self.state, comPosRef, nu
            )
            comPosTrack = crocoddyl.CostModelResidual(self.state, comPosResidual)
            costModel.addCost("comPosTrack", comPosTrack, 1e6)

        # Cost of tracking body pose reference (if needed)
        if bodyPoseRef is not None:
            bodyPoseResidual = crocoddyl.ResidualModelFramePlacement(
                self.state, self.bodyId, bodyPoseRef, nu
            )
            bodyPoseTrack = crocoddyl.CostModelResidual(self.state, bodyPoseResidual)
            costModel.addCost("bodyPoseTrack", bodyPoseTrack, 1e6)

        # Cost of tracking swing trajectories (if needed)
        if swingFootPoseRef is not None:
            for i in swingFootPoseRef:
                footId = i[0]
                footPoseRef = i[1]
                footPoseResidual = crocoddyl.ResidualModelFramePlacement(
                    self.state, footId, footPoseRef, nu
                )
                footPoseTrack = crocoddyl.CostModelResidual(
                    self.state, footPoseResidual
                )
                costModel.addCost(
                    self.rmodel.frames[footId].name + "_footPoseTrack",
                    footPoseTrack,
                    1e6,
                )

        # Cost of contact wrench cone
        for i in supportFootIds:
            # cone = crocoddyl.WrenchCone(self.Rsurf, self.mu, np.array([0.1, 0.05])) # for talos
            cone = crocoddyl.WrenchCone(
                self.Rsurf, self.mu, np.array([0.05, 0.05])
            )  # for wheeled biped
            wrenchResidual = crocoddyl.ResidualModelContactWrenchCone(
                self.state, i, cone, nu, self._fwddyn
            )
            wrenchActivation = crocoddyl.ActivationModelQuadraticBarrier(
                crocoddyl.ActivationBounds(cone.lb, cone.ub)
            )
            wrenchCone = crocoddyl.CostModelResidual(
                self.state, wrenchActivation, wrenchResidual
            )
            costModel.addCost(
                self.rmodel.frames[i].name + "_wrenchCone", wrenchCone, 1e1
            )

        # Cost of state regularization ("half_sitting" pose)
        stateWeights = np.array(
            [0.0] * 3
            + [500.0, 500.0, 0.0]
            + [0.01] * (self.state.nv - 6)
            + [10] * self.state.nv
        )
        stateResidual = crocoddyl.ResidualModelState(
            self.state, self.rmodel.defaultState, nu
        )
        stateActivation = crocoddyl.ActivationModelWeightedQuad(stateWeights**2)
        stateReg = crocoddyl.CostModelResidual(
            self.state, stateActivation, stateResidual
        )
        costModel.addCost("stateReg", stateReg, 1e2)

        # Cost of control regularization
        if self._fwddyn:
            ctrlResidual = crocoddyl.ResidualModelControl(self.state, nu)
        else:
            ctrlResidual = crocoddyl.ResidualModelJointEffort(
                self.state, self.actuation, nu
            )
        ctrlReg = crocoddyl.CostModelResidual(self.state, ctrlResidual)
        costModel.addCost("ctrlReg", ctrlReg, 1e-3)

        # Creating the action model for the KKT dynamics with simpletic Euler integration scheme
        if self._fwddyn:
            dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(
                self.state, self.actuation, contactModel, costModel, 0.0, True
            )
        else:
            dmodel = crocoddyl.DifferentialActionModelContactInvDynamics(
                self.state, self.actuation, contactModel, costModel
            )

        # Represent control input over the duration of dt
        if self._control == "one":
            control = crocoddyl.ControlParametrizationModelPolyOne(nu)
        elif self._control == "rk4":
            control = crocoddyl.ControlParametrizationModelPolyTwoRK(
                nu, crocoddyl.RKType.four
            )
        elif self._control == "rk3":
            control = crocoddyl.ControlParametrizationModelPolyTwoRK(
                nu, crocoddyl.RKType.three
            )
        else:
            control = crocoddyl.ControlParametrizationModelPolyZero(nu)

        # Discretization method
        if self._integrator == "euler":
            model = crocoddyl.IntegratedActionModelEuler(dmodel, control, timeStep)
        elif self._integrator == "rk4":
            model = crocoddyl.IntegratedActionModelRK(
                dmodel, control, crocoddyl.RKType.four, timeStep
            )
        elif self._integrator == "rk3":
            model = crocoddyl.IntegratedActionModelRK(
                dmodel, control, crocoddyl.RKType.three, timeStep
            )
        elif self._integrator == "rk2":
            model = crocoddyl.IntegratedActionModelRK(
                dmodel, control, crocoddyl.RKType.two, timeStep
            )
        else:
            model = crocoddyl.IntegratedActionModelEuler(dmodel, control, timeStep)

        return model

    def createFootSwitchModel(self, supportFootIds, swingFootTask, pseudoImpulse=True):
        """Action model for a foot switch phase.

        :param supportFootIds: Ids of the constrained feet
        :param swingFootTask: swinging foot task
        :param pseudoImpulse: true for pseudo-impulse models, otherwise it uses the
            impulse model
        :return action model for a foot switch phase
        """
        if pseudoImpulse:
            return self.createPseudoImpulseModel(supportFootIds, swingFootTask)
        else:
            return self.createImpulseModel(supportFootIds, swingFootTask)

    def createPseudoImpulseModel(self, supportFootIds, swingFootTask):
        """Action model for pseudo-impulse models.

        A pseudo-impulse model consists of adding high-penalty cost for the contact
        velocities.
        :param swingFootTask: swinging foot task
        :return pseudo-impulse differential action model
        """
        # Creating a 6D multi-contact model, and then including the supporting
        # foot
        if self._fwddyn:
            nu = self.actuation.nu
        else:
            nu = self.state.nv + 6 * len(supportFootIds)
        contactModel = crocoddyl.ContactModelMultiple(self.state, nu)
        for i in supportFootIds:
            supportContactModel = crocoddyl.ContactModel6D(
                self.state,
                i,
                pinocchio.SE3.Identity(),
                pinocchio.LOCAL_WORLD_ALIGNED,
                nu,
                np.array([0.0, 50.0]),
            )
            contactModel.addContact(
                self.rmodel.frames[i].name + "_contact", supportContactModel
            )
        # Creating the cost model for a contact phase
        costModel = crocoddyl.CostModelSum(self.state, nu)
        for i in supportFootIds:
            cone = crocoddyl.WrenchCone(self.Rsurf, self.mu, np.array([0.1, 0.05]))
            wrenchResidual = crocoddyl.ResidualModelContactWrenchCone(
                self.state, i, cone, nu, self._fwddyn
            )
            wrenchActivation = crocoddyl.ActivationModelQuadraticBarrier(
                crocoddyl.ActivationBounds(cone.lb, cone.ub)
            )
            wrenchCone = crocoddyl.CostModelResidual(
                self.state, wrenchActivation, wrenchResidual
            )
            costModel.addCost(
                self.rmodel.frames[i].name + "_wrenchCone", wrenchCone, 1e1
            )
        if swingFootTask is not None:
            for i in swingFootTask:
                framePlacementResidual = crocoddyl.ResidualModelFramePlacement(
                    self.state, i[0], i[1], nu
                )
                frameVelocityResidual = crocoddyl.ResidualModelFrameVelocity(
                    self.state,
                    i[0],
                    pinocchio.Motion.Zero(),
                    pinocchio.LOCAL_WORLD_ALIGNED,
                    nu,
                )
                footTrack = crocoddyl.CostModelResidual(
                    self.state, framePlacementResidual
                )
                impulseFootVelCost = crocoddyl.CostModelResidual(
                    self.state, frameVelocityResidual
                )
                costModel.addCost(
                    self.rmodel.frames[i[0]].name + "_footTrack", footTrack, 1e8
                )
                costModel.addCost(
                    self.rmodel.frames[i[0]].name + "_impulseVel",
                    impulseFootVelCost,
                    1e6,
                )
        stateWeights = np.array(
            [0.0] * 3
            + [500.0] * 3
            + [0.01] * (self.state.nv - 6)
            + [10] * self.state.nv
        )
        stateResidual = crocoddyl.ResidualModelState(
            self.state, self.rmodel.defaultState, nu
        )
        stateActivation = crocoddyl.ActivationModelWeightedQuad(stateWeights**2)
        stateReg = crocoddyl.CostModelResidual(
            self.state, stateActivation, stateResidual
        )
        if self._fwddyn:
            ctrlResidual = crocoddyl.ResidualModelControl(self.state, nu)
        else:
            ctrlResidual = crocoddyl.ResidualModelJointEffort(
                self.state, self.actuation, nu
            )
        ctrlReg = crocoddyl.CostModelResidual(self.state, ctrlResidual)
        costModel.addCost("stateReg", stateReg, 1e1)
        costModel.addCost("ctrlReg", ctrlReg, 1e-3)
        # Creating the action model for the KKT dynamics with simpletic Euler
        # integration scheme
        if self._fwddyn:
            dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(
                self.state, self.actuation, contactModel, costModel, 0.0, True
            )
        else:
            dmodel = crocoddyl.DifferentialActionModelContactInvDynamics(
                self.state, self.actuation, contactModel, costModel
            )
        if self._integrator == "euler":
            model = crocoddyl.IntegratedActionModelEuler(dmodel, 0.0)
        elif self._integrator == "rk4":
            model = crocoddyl.IntegratedActionModelRK(
                dmodel, crocoddyl.RKType.four, 0.0
            )
        elif self._integrator == "rk3":
            model = crocoddyl.IntegratedActionModelRK(
                dmodel, crocoddyl.RKType.three, 0.0
            )
        elif self._integrator == "rk2":
            model = crocoddyl.IntegratedActionModelRK(dmodel, crocoddyl.RKType.two, 0.0)
        else:
            model = crocoddyl.IntegratedActionModelEuler(dmodel, 0.0)
        return model

    def createImpulseModel(
        self, supportFootIds, swingFootTask, JMinvJt_damping=0.0, r_coeff=0.0
    ):
        """Action model for impulse models.

        An impulse model consists of describing the impulse dynamics against a set of
        contacts.
        :param supportFootIds: Ids of the constrained feet
        :param swingFootTask: swinging foot task
        :return impulse action model
        """
        # Creating a 6D multi-contact model, and then including the supporting foot
        impulseModel = crocoddyl.ImpulseModelMultiple(self.state)
        for i in supportFootIds:
            supportContactModel = crocoddyl.ImpulseModel6D(
                self.state, i, pinocchio.LOCAL_WORLD_ALIGNED
            )
            impulseModel.addImpulse(
                self.rmodel.frames[i].name + "_impulse", supportContactModel
            )
        # Creating the cost model for a contact phase
        costModel = crocoddyl.CostModelSum(self.state, 0)
        if swingFootTask is not None:
            for i in swingFootTask:
                framePlacementResidual = crocoddyl.ResidualModelFramePlacement(
                    self.state, i[0], i[1], 0
                )
                footTrack = crocoddyl.CostModelResidual(
                    self.state, framePlacementResidual
                )
                costModel.addCost(
                    self.rmodel.frames[i[0]].name + "_footTrack", footTrack, 1e8
                )
        stateWeights = np.array(
            [1.0] * 6 + [0.1] * (self.rmodel.nv - 6) + [10] * self.rmodel.nv
        )
        stateResidual = crocoddyl.ResidualModelState(
            self.state, self.rmodel.defaultState, 0
        )
        stateActivation = crocoddyl.ActivationModelWeightedQuad(stateWeights**2)
        stateReg = crocoddyl.CostModelResidual(
            self.state, stateActivation, stateResidual
        )
        costModel.addCost("stateReg", stateReg, 1e1)
        # Creating the action model for the KKT dynamics with simpletic Euler
        # integration scheme
        model = crocoddyl.ActionModelImpulseFwdDynamics(
            self.state, impulseModel, costModel
        )
        model.JMinvJt_damping = JMinvJt_damping
        model.r_coeff = r_coeff
        return model


def plotSolution(solver, bounds=True, figIndex=1, figTitle="", show=True):
    import matplotlib.pyplot as plt

    xs, us, cs = [], [], []
    if bounds:
        us_lb, us_ub = [], []
        xs_lb, xs_ub = [], []

    def updateTrajectories(solver):
        xs.extend(solver.xs[:-1])
        for m, d in zip(solver.problem.runningModels, solver.problem.runningDatas):
            if hasattr(m, "differential"):
                cs.append(d.differential.multibody.pinocchio.com[0])
                us.append(d.differential.multibody.joint.tau)
                if bounds and isinstance(
                    m.differential, crocoddyl.DifferentialActionModelContactFwdDynamics
                ):
                    us_lb.extend([m.u_lb])
                    us_ub.extend([m.u_ub])
            else:
                cs.append(d.multibody.pinocchio.com[0])
                us.append(np.zeros(nu))
                if bounds:
                    us_lb.append(np.nan * np.ones(nu))
                    us_ub.append(np.nan * np.ones(nu))
            if bounds:
                xs_lb.extend([m.state.lb])
                xs_ub.extend([m.state.ub])

    if isinstance(solver, list):
        for s in solver:
            rmodel = solver[0].problem.runningModels[0].state.pinocchio
            nq, nv, nu = (
                rmodel.nq,
                rmodel.nv,
                solver[0].problem.runningModels[0].differential.actuation.nu,
            )
            updateTrajectories(s)
    else:
        rmodel = solver.problem.runningModels[0].state.pinocchio
        nq, nv, nu = (
            rmodel.nq,
            rmodel.nv,
            solver.problem.runningModels[0].differential.actuation.nu,
        )
        updateTrajectories(solver)

    # Getting the state and control trajectories
    nx = nq + nv
    X = [0.0] * nx
    U = [0.0] * nu
    if bounds:
        U_LB = [0.0] * nu
        U_UB = [0.0] * nu
        X_LB = [0.0] * nx
        X_UB = [0.0] * nx
    for i in range(nx):
        X[i] = [x[i] for x in xs]
        if bounds:
            X_LB[i] = [x[i] for x in xs_lb]
            X_UB[i] = [x[i] for x in xs_ub]
    for i in range(nu):
        U[i] = [u[i] for u in us]
        if bounds:
            U_LB[i] = [u[i] for u in us_lb]
            U_UB[i] = [u[i] for u in us_ub]

    # Plotting the joint positions, velocities and torques
    plt.figure(figIndex)
    plt.suptitle(figTitle)
    legJointNames = ["1", "2", "3", "4", "5", "6"]
    # left foot
    plt.subplot(2, 3, 1)
    plt.title("joint position [rad]")
    [plt.plot(X[k], label=legJointNames[i]) for i, k in enumerate(range(7, 13))]
    if bounds:
        [plt.plot(X_LB[k], "--r") for i, k in enumerate(range(7, 13))]
        [plt.plot(X_UB[k], "--r") for i, k in enumerate(range(7, 13))]
    plt.ylabel("LF")
    plt.legend()
    plt.subplot(2, 3, 2)
    plt.title("joint velocity [rad/s]")
    [
        plt.plot(X[k], label=legJointNames[i])
        for i, k in enumerate(range(nq + 6, nq + 12))
    ]
    if bounds:
        [plt.plot(X_LB[k], "--r") for i, k in enumerate(range(nq + 6, nq + 12))]
        [plt.plot(X_UB[k], "--r") for i, k in enumerate(range(nq + 6, nq + 12))]
    plt.ylabel("LF")
    plt.legend()
    plt.subplot(2, 3, 3)
    plt.title("joint torque [Nm]")
    [plt.plot(U[k], label=legJointNames[i]) for i, k in enumerate(range(0, 6))]
    if bounds:
        [plt.plot(U_LB[k], "--r") for i, k in enumerate(range(0, 6))]
        [plt.plot(U_UB[k], "--r") for i, k in enumerate(range(0, 6))]
    plt.ylabel("LF")
    plt.legend()

    # right foot
    plt.subplot(2, 3, 4)
    [plt.plot(X[k], label=legJointNames[i]) for i, k in enumerate(range(13, 19))]
    if bounds:
        [plt.plot(X_LB[k], "--r") for i, k in enumerate(range(13, 19))]
        [plt.plot(X_UB[k], "--r") for i, k in enumerate(range(13, 19))]
    plt.ylabel("RF")
    plt.xlabel("knots")
    plt.legend()
    plt.subplot(2, 3, 5)
    [
        plt.plot(X[k], label=legJointNames[i])
        for i, k in enumerate(range(nq + 12, nq + 18))
    ]
    if bounds:
        [plt.plot(X_LB[k], "--r") for i, k in enumerate(range(nq + 12, nq + 18))]
        [plt.plot(X_UB[k], "--r") for i, k in enumerate(range(nq + 12, nq + 18))]
    plt.ylabel("RF")
    plt.xlabel("knots")
    plt.legend()
    plt.subplot(2, 3, 6)
    [plt.plot(U[k], label=legJointNames[i]) for i, k in enumerate(range(6, 12))]
    if bounds:
        [plt.plot(U_LB[k], "--r") for i, k in enumerate(range(6, 12))]
        [plt.plot(U_UB[k], "--r") for i, k in enumerate(range(6, 12))]
    plt.ylabel("RF")
    plt.xlabel("knots")
    plt.legend()

    plt.figure(figIndex + 1)
    rdata = rmodel.createData()
    Cx = []
    Cy = []
    for x in xs:
        q = x[: rmodel.nq]
        c = pinocchio.centerOfMass(rmodel, rdata, q)
        Cx.append(c[0])
        Cy.append(c[1])
    plt.plot(Cx, Cy)
    plt.title("CoM position")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.grid(True)
    if show:
        plt.show()
