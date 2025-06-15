import numpy as np
import pinocchio
import crocoddyl
from typing import List, Tuple


# Reference:
# https://github.com/loco-3d/crocoddyl/blob/devel/bindings/python/crocoddyl/utils/biped.py
class JumpingProblem:
    def __init__(
        self,
        robot_model: pinocchio.Model,
        body_frame_name: str,
        rf_contact_frame_name: str,
        lf_contact_frame_name: str,
        integrator: str = "rk4",
        control: str = "zero",
    ):
        """
        Construct a jumping problem.

        :param robot_model: Pinocchio robot model.
        :param body_frame_name: Name of the body frame.
        :param rf_contact_frame_name: Name of the right foot contact frame.
        :param lf_contact_frame_name: Name of the left foot contact frame.
        :param integrator: Type of the integrator.
        :param control: Type of the control parametrization.
        """
        self.robot_model = robot_model
        self.robot_data = robot_model.createData()
        self.state = crocoddyl.StateMultibody(self.robot_model)
        self.actuation = crocoddyl.ActuationModelFloatingBase(self.state)
        self._integrator = integrator
        self._control = control

        # Get frame IDs of base and foot contact frames
        self.body_frame_id = self.robot_model.getFrameId(body_frame_name)
        self.rf_contact_frame_id = self.robot_model.getFrameId(rf_contact_frame_name)
        self.lf_contact_frame_id = self.robot_model.getFrameId(lf_contact_frame_name)

        # Get frame IDs of base and foot contact frames
        q0 = self.robot_model.referenceConfigurations["half_sitting"]
        self.robot_model.defaultState = np.concatenate(
            [q0, np.zeros(self.robot_model.nv)]
        )

        # Define friction coefficient and ground
        self.mu = 0.7
        self.R_ground = np.eye(3)

    def create_jumping_problem(
        self,
        x0: np.ndarray,
        jump_height: float,
        jump_length: List[float],
        jump_yaw_rot_deg: float,
        dt: float,
        num_ground_knots: int,
        num_flying_knots: int,
    ) -> crocoddyl.ShootingProblem:
        """
        Construct a shooting problem for jumping.

        :param x0: Initial state of the robot (q0, v0).
        :param jump_height: Height of the jump in meters.
        :param jump_length: Length of the jump in meters [x, y, z] (global frame).
        :param jump_yaw_rot_deg: Yaw rotation during the jump in degrees (body frame).
        :param dt: Time step length in seconds.
        :param num_ground_knots: Number of knots on the ground.
        :param num_flying_knots: Number of knots in the air.

        :return: A shooting problem for jumping.
        """
        jump_yaw_rot = np.deg2rad(jump_yaw_rot_deg)
        jump_action_models = []

        # Get initial foot pose
        q0 = x0[: self.robot_model.nq]
        pinocchio.forwardKinematics(self.robot_model, self.robot_data, q0)
        pinocchio.updateFramePlacements(self.robot_model, self.robot_data)
        rf_contact_pos_0 = self.robot_data.oMf[self.rf_contact_frame_id].translation
        lf_contact_pos_0 = self.robot_data.oMf[self.lf_contact_frame_id].translation

        # Get initial body frame pose (global frame)
        body_pos_0 = self.robot_data.oMf[self.body_frame_id].translation
        body_rot_0 = self.robot_data.oMf[self.body_frame_id].rotation

        # Set initial body_frame position ref (global frame)
        body_pos_ref_0 = (rf_contact_pos_0 + lf_contact_pos_0) / 2
        body_pos_ref_0[2] = body_pos_0[2]

        body_to_rf_contact_pos_0 = rf_contact_pos_0 - body_pos_0
        body_to_lf_contact_pos_0 = lf_contact_pos_0 - body_pos_0

        # ---------------------------------------------------------------------------- #
        # Take-Off Phase ------------------------------------------------------------- #
        # ---------------------------------------------------------------------------- #
        take_off = [
            self.create_knot_action_model(
                dt,
                [self.lf_contact_frame_id, self.rf_contact_frame_id],
            )
            for _ in range(num_ground_knots)
        ]

        # ---------------------------------------------------------------------------- #
        # Flying-Up Phase ------------------------------------------------------------ #
        # ---------------------------------------------------------------------------- #
        fly_up = []
        last_body_pos_ref = np.zeros(3)
        last_body_yaw_ref = 0.0
        for k in range(num_flying_knots):
            # Body position reference
            body_pos_ref = (
                np.array(
                    [
                        jump_length[0] / 2.0,
                        jump_length[1] / 2.0,
                        jump_length[2] / 2.0 + jump_height,
                    ]
                )
                * (k + 1)
                / num_flying_knots
                + body_pos_ref_0
            )
            last_body_pos_ref = body_pos_ref

            # Body Yaw reference
            body_yaw_ref = jump_yaw_rot / 2.0 * (k + 1) / num_flying_knots
            last_body_yaw_ref = body_yaw_ref

            # Body SE3 reference
            body_pose_ref = pinocchio.SE3(
                body_rot_0 @ pinocchio.rpy.rpyToMatrix(0.0, 0.0, body_yaw_ref),
                body_pos_ref,
            )

            # Create knot action models
            fly_up.append(
                self.create_knot_action_model(dt, [], body_pose_ref=body_pose_ref)
            )

        # ---------------------------------------------------------------------------- #
        # Flying-Down Phase ---------------------------------------------------------- #
        # ---------------------------------------------------------------------------- #
        fly_down = []
        for k in range(num_flying_knots):
            # Body position reference
            body_pos_ref = (
                np.array(
                    [
                        jump_length[0] / 2.0,
                        jump_length[1] / 2.0,
                        jump_length[2] / 2.0 - jump_height,
                    ]
                )
                * (k + 1)
                / num_flying_knots
                + last_body_pos_ref
            )

            # Body Yaw reference
            body_yaw_ref = (
                last_body_yaw_ref + jump_yaw_rot / 2.0 * (k + 1) / num_flying_knots
            )

            # Body SE3 reference
            body_pose_ref = pinocchio.SE3(
                body_rot_0 @ pinocchio.rpy.rpyToMatrix(0.0, 0.0, body_yaw_ref),
                body_pos_ref,
            )

            # Create the action model and append
            fly_down.append(
                self.create_knot_action_model(dt, [], body_pose_ref=body_pose_ref)
            )

        # ---------------------------------------------------------------------------- #
        # Landing Phase -------------------------------------------------------------- #
        # ---------------------------------------------------------------------------- #
        rot_ref_final = body_rot_0 @ pinocchio.rpy.rpyToMatrix(0.0, 0.0, jump_yaw_rot)
        foot_pose_ref_final = [
            [
                self.lf_contact_frame_id,
                pinocchio.SE3(
                    rot_ref_final,
                    body_pos_0
                    + jump_length
                    + pinocchio.rpy.rpyToMatrix(0.0, 0.0, jump_yaw_rot)
                    @ body_to_lf_contact_pos_0,
                ),
            ],
            [
                self.rf_contact_frame_id,
                pinocchio.SE3(
                    rot_ref_final,
                    body_pos_0
                    + jump_length
                    + pinocchio.rpy.rpyToMatrix(0.0, 0.0, jump_yaw_rot)
                    @ body_to_rf_contact_pos_0,
                ),
            ],
        ]
        landing = [
            self.create_foot_switch_model(
                [self.lf_contact_frame_id, self.rf_contact_frame_id],
                foot_pose_ref_final,
                False,
            )
        ]
        landed = [
            self.create_knot_action_model(
                dt,
                [self.lf_contact_frame_id, self.rf_contact_frame_id],
                body_pose_ref=pinocchio.SE3(rot_ref_final, body_pos_0 + jump_length),
            )
            for _ in range(int(num_ground_knots / 2))
        ]

        # ---------------------------------------------------------------------------- #
        # Create Shooting Problem ---------------------------------------------------- #
        # ---------------------------------------------------------------------------- #
        jump_action_models += take_off
        jump_action_models += fly_up
        jump_action_models += fly_down
        jump_action_models += landing
        jump_action_models += landed
        return crocoddyl.ShootingProblem(
            x0, jump_action_models[:-1], jump_action_models[-1]
        )

    def create_knot_action_model(
        self,
        dt: float,
        support_foot_ids: List[int],
        com_pos_ref: np.ndarray = None,
        body_pose_ref: pinocchio.SE3 = None,
        swing_foot_pose_ref: pinocchio.SE3 = None,
    ) -> crocoddyl.IntegratedActionModelAbstract:
        """
        Create an action model for one knot.

        :param dt: Time step length in seconds.
        :param support_foot_ids: IDs of support feet.
        :param com_pos_ref: Reference CoM position (global frame).
        :param body_pose_ref: Reference body frame pose (global frame).
        :param swing_foot_pose_ref: Reference swing foot pose (global frame).

        :return: An action model for one knot.
        """
        nu = self.actuation.nu
        costs = crocoddyl.CostModelSum(self.state, nu)

        # ---------------------------------------------------------------------------- #
        # Contact models for support feet -------------------------------------------- #
        # ---------------------------------------------------------------------------- #
        contact_models = crocoddyl.ContactModelMultiple(self.state, nu)
        for foot_id in support_foot_ids:
            support_contact_model = crocoddyl.ContactModel6D(
                self.state,
                foot_id,
                pinocchio.SE3.Identity(),
                pinocchio.LOCAL_WORLD_ALIGNED,
                nu,
                np.array([0.0, 30.0]),
            )
            contact_models.addContact(
                self.robot_model.frames[foot_id].name + "_contact",
                support_contact_model,
            )

        # ---------------------------------------------------------------------------- #
        # Contact wrench cone costs -------------------------------------------------- #
        # ---------------------------------------------------------------------------- #
        for i in support_foot_ids:
            cone = crocoddyl.WrenchCone(self.R_ground, self.mu, np.array([0.1, 0.05]))
            wrench_residual = crocoddyl.ResidualModelContactWrenchCone(
                self.state, i, cone, nu, fwddyn=True
            )
            wrench_activation = crocoddyl.ActivationModelQuadraticBarrier(
                crocoddyl.ActivationBounds(cone.lb, cone.ub)
            )
            wrench_cone = crocoddyl.CostModelResidual(
                self.state, wrench_activation, wrench_residual
            )
            costs.addCost(
                self.robot_model.frames[i].name + "_wrenchCone", wrench_cone, 1e1
            )

        # ---------------------------------------------------------------------------- #
        # CoM position tracking cost ------------------------------------------------- #
        # ---------------------------------------------------------------------------- #
        if com_pos_ref is not None:
            com_pos_residual = crocoddyl.ResidualModelCoMPosition(
                self.state, com_pos_ref, nu
            )
            com_pos_track = crocoddyl.CostModelResidual(self.state, com_pos_residual)
            costs.addCost("com_pos_track", com_pos_track, 1e6)

        # ---------------------------------------------------------------------------- #
        # Body pose tracking cost ---------------------------------------------------- #
        # ---------------------------------------------------------------------------- #
        if body_pose_ref is not None:
            body_pose_residual = crocoddyl.ResidualModelFramePlacement(
                self.state, self.body_frame_id, body_pose_ref, nu
            )
            body_pose_track = crocoddyl.CostModelResidual(
                self.state, body_pose_residual
            )
            costs.addCost("body_pose_track", body_pose_track, 1e6)

        # ---------------------------------------------------------------------------- #
        # Swing foot pose tracking cost ---------------------------------------------- #
        # ---------------------------------------------------------------------------- #
        if swing_foot_pose_ref is not None:
            for i in swing_foot_pose_ref:
                foot_id = i[0]
                foot_pose_ref = i[1]
                foot_pose_residual = crocoddyl.ResidualModelFramePlacement(
                    self.state, foot_id, foot_pose_ref, nu
                )
                foot_pose_track = crocoddyl.CostModelResidual(
                    self.state, foot_pose_residual
                )
                costs.addCost(
                    self.robot_model.frames[foot_id].name + "_footPoseTrack",
                    foot_pose_track,
                    1e6,
                )

        # ---------------------------------------------------------------------------- #
        # Whole-body joint state tracking cost --------------------------------------- #
        # ---------------------------------------------------------------------------- #
        x_track_weights = np.array(
            [0.0] * 3
            + [500.0, 500.0, 0.0]
            + [0.01] * (self.state.nv - 6)
            + [10] * self.state.nv
        )
        x_track_residual = crocoddyl.ResidualModelState(
            self.state, self.robot_model.defaultState, nu
        )
        x_track_activation = crocoddyl.ActivationModelWeightedQuad(x_track_weights**2)
        x_track_cost = crocoddyl.CostModelResidual(
            self.state, x_track_activation, x_track_residual
        )
        costs.addCost("x_track_cost", x_track_cost, 1e2)

        # ---------------------------------------------------------------------------- #
        # Control regularization cost ------------------------------------------------ #
        # ---------------------------------------------------------------------------- #
        ctrl_residual = crocoddyl.ResidualModelControl(self.state, nu)
        ctrl_regu_cost = crocoddyl.CostModelResidual(self.state, ctrl_residual)
        costs.addCost("ctrl_regu_cost", ctrl_regu_cost, 1e-3)

        # ---------------------------------------------------------------------------- #
        # Create forward dynamics model ---------------------------------------------- #
        # ---------------------------------------------------------------------------- #
        dyn_model = crocoddyl.DifferentialActionModelContactFwdDynamics(
            self.state, self.actuation, contact_models, costs, 0.0, True
        )

        # ---------------------------------------------------------------------------- #
        # Control parametrization ---------------------------------------------------- #
        # ---------------------------------------------------------------------------- #
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

        # ---------------------------------------------------------------------------- #
        # Discretize dynamics model -------------------------------------------------- #
        # ---------------------------------------------------------------------------- #
        if self._integrator == "euler":
            model = crocoddyl.IntegratedActionModelEuler(dyn_model, control, dt)
        elif self._integrator == "rk4":
            model = crocoddyl.IntegratedActionModelRK(
                dyn_model, control, crocoddyl.RKType.four, dt
            )
        elif self._integrator == "rk3":
            model = crocoddyl.IntegratedActionModelRK(
                dyn_model, control, crocoddyl.RKType.three, dt
            )
        elif self._integrator == "rk2":
            model = crocoddyl.IntegratedActionModelRK(
                dyn_model, control, crocoddyl.RKType.two, dt
            )
        else:
            model = crocoddyl.IntegratedActionModelEuler(dyn_model, control, dt)
        return model

    def create_foot_switch_model(
        self,
        support_foot_ids: List[int],
        swing_foot_pos_ref: List[Tuple[int, pinocchio.SE3]],
        pseudo_impulse: bool = False,
    ) -> crocoddyl.IntegratedActionModelAbstract:
        """
        Create an action model for foot mode switch (impulse dynamics).

        :param support_foot_ids: IDs of support feet.
        :param swing_foot_pos_ref: Reference swing foot poses (global frame).
        :param pseudo_impulse: If True, use pseudo-impulse model; otherwise, use impulse model.

        :return: An action model for foot mode switch (impulse dynamics).
        """
        if pseudo_impulse:
            return self.create_pseudo_impulse_model(
                support_foot_ids, swing_foot_pos_ref
            )
        else:
            return self.create_impulse_model(support_foot_ids, swing_foot_pos_ref)

    def create_pseudo_impulse_model(
        self,
        support_foot_ids: List[int],
        swing_foot_pos_ref: List[Tuple[int, pinocchio.SE3]] = None,
    ) -> crocoddyl.IntegratedActionModelAbstract:
        """
        Create an action model for pseudo-impulse dynamics. This is used to handle the mode
        switch caused by contacts.

        :param support_foot_ids: IDs of support feet.
        :param swing_foot_pos_ref: Reference swing foot poses (global frame).

        :return: A pseudo-impulse model for foot mode switch.
        """
        nu = self.actuation.nu
        costs = crocoddyl.CostModelSum(self.state, nu)
        contact_models = crocoddyl.ContactModelMultiple(self.state, nu)

        # ---------------------------------------------------------------------------- #
        # Contact models for support feet -------------------------------------------- #
        # ---------------------------------------------------------------------------- #
        for i in support_foot_ids:
            support_contact_model = crocoddyl.ContactModel6D(
                self.state,
                i,
                pinocchio.SE3.Identity(),
                pinocchio.LOCAL_WORLD_ALIGNED,
                nu,
                np.array([0.0, 50.0]),
            )
            contact_models.addContact(
                self.robot_model.frames[i].name + "_contact", support_contact_model
            )

        # ---------------------------------------------------------------------------- #
        # Contact wrench cone costs -------------------------------------------------- #
        # ---------------------------------------------------------------------------- #
        for i in support_foot_ids:
            cone = crocoddyl.WrenchCone(self.R_ground, self.mu, np.array([0.1, 0.05]))
            wrench_residual = crocoddyl.ResidualModelContactWrenchCone(
                self.state, i, cone, nu, fwddyn=True
            )
            wrench_activation = crocoddyl.ActivationModelQuadraticBarrier(
                crocoddyl.ActivationBounds(cone.lb, cone.ub)
            )
            wrench_cone = crocoddyl.CostModelResidual(
                self.state, wrench_activation, wrench_residual
            )
            costs.addCost(
                self.robot_model.frames[i].name + "_wrenchCone", wrench_cone, 1e1
            )

        # ---------------------------------------------------------------------------- #
        # Swing foot pose & velocity tracking cost ----------------------------------- #
        # ---------------------------------------------------------------------------- #
        if swing_foot_pos_ref is not None:
            for i in swing_foot_pos_ref:
                frame_placement_residual = crocoddyl.ResidualModelFramePlacement(
                    self.state, i[0], i[1], nu
                )
                frame_velocity_residual = crocoddyl.ResidualModelFrameVelocity(
                    self.state,
                    i[0],
                    pinocchio.Motion.Zero(),
                    pinocchio.LOCAL_WORLD_ALIGNED,
                    nu,
                )
                foot_track = crocoddyl.CostModelResidual(
                    self.state, frame_placement_residual
                )
                impulse_foot_vel_cost = crocoddyl.CostModelResidual(
                    self.state, frame_velocity_residual
                )
                costs.addCost(
                    self.robot_model.frames[i[0]].name + "_footTrack", foot_track, 1e8
                )
                costs.addCost(
                    self.robot_model.frames[i[0]].name + "_impulseVel",
                    impulse_foot_vel_cost,
                    1e6,
                )

        # ---------------------------------------------------------------------------- #
        # Whole-body joint state tracking cost --------------------------------------- #
        # ---------------------------------------------------------------------------- #
        x_track_weights = np.array(
            [0.0] * 3
            + [500.0] * 3
            + [0.01] * (self.state.nv - 6)
            + [10] * self.state.nv
        )
        x_track_residual = crocoddyl.ResidualModelState(
            self.state, self.robot_model.defaultState, nu
        )
        x_track_activation = crocoddyl.ActivationModelWeightedQuad(x_track_weights**2)
        x_track_cost = crocoddyl.CostModelResidual(
            self.state, x_track_activation, x_track_residual
        )

        # ---------------------------------------------------------------------------- #
        # Control regularization cost ------------------------------------------------ #
        # ---------------------------------------------------------------------------- #
        ctrl_residual = crocoddyl.ResidualModelControl(self.state, nu)
        ctrl_regu_cost = crocoddyl.CostModelResidual(self.state, ctrl_residual)
        costs.addCost("x_track_cost", x_track_cost, 1e1)
        costs.addCost("ctrl_regu_cost", ctrl_regu_cost, 1e-3)

        # ---------------------------------------------------------------------------- #
        # Create forward dynamics model ---------------------------------------------- #
        # ---------------------------------------------------------------------------- #
        dyn_model = crocoddyl.DifferentialActionModelContactFwdDynamics(
            self.state, self.actuation, contact_models, costs, 0.0, True
        )

        # ---------------------------------------------------------------------------- #
        # Discretize dynamics model -------------------------------------------------- #
        # ---------------------------------------------------------------------------- #
        if self._integrator == "euler":
            model = crocoddyl.IntegratedActionModelEuler(dyn_model, 0.0)
        elif self._integrator == "rk4":
            model = crocoddyl.IntegratedActionModelRK(
                dyn_model, crocoddyl.RKType.four, 0.0
            )
        elif self._integrator == "rk3":
            model = crocoddyl.IntegratedActionModelRK(
                dyn_model, crocoddyl.RKType.three, 0.0
            )
        elif self._integrator == "rk2":
            model = crocoddyl.IntegratedActionModelRK(
                dyn_model, crocoddyl.RKType.two, 0.0
            )
        else:
            model = crocoddyl.IntegratedActionModelEuler(dyn_model, 0.0)
        return model

    def create_impulse_model(
        self,
        support_foot_ids: List[int],
        swing_foot_pos_ref: List[Tuple[int, pinocchio.SE3]] = None,
        r_coeff: float = 0.0,
        JMinvJt_damping: float = 1e-12,
    ) -> crocoddyl.IntegratedActionModelAbstract:
        """
        Create an action model for impulse dynamics. This is used to handle the mode
        switch caused by contacts.

        :param support_foot_ids: IDs of support feet.
        :param swing_foot_pos_ref: Reference swing foot poses (global frame).
        :param r_coeff: Restitution coefficient.
        :param JMinvJt_damping: Damping term used in operational space inertia matrix.

        :return: An impulse model for foot mode switch.
        """
        costs = crocoddyl.CostModelSum(self.state, 0)

        # ---------------------------------------------------------------------------- #
        # Impulse models for support feet -------------------------------------------- #
        # ---------------------------------------------------------------------------- #
        impulse_model = crocoddyl.ImpulseModelMultiple(self.state)
        for i in support_foot_ids:
            support_contact_model = crocoddyl.ImpulseModel6D(
                self.state, i, pinocchio.LOCAL_WORLD_ALIGNED
            )
            impulse_model.addImpulse(
                self.robot_model.frames[i].name + "_impulse", support_contact_model
            )

        # ---------------------------------------------------------------------------- #
        # Foot poses tracking costs -------------------------------------------------- #
        # ---------------------------------------------------------------------------- #
        if swing_foot_pos_ref is not None:
            for i in swing_foot_pos_ref:
                frame_placement_residual = crocoddyl.ResidualModelFramePlacement(
                    self.state, i[0], i[1], 0
                )
                foot_track = crocoddyl.CostModelResidual(
                    self.state, frame_placement_residual
                )
                costs.addCost(
                    self.robot_model.frames[i[0]].name + "_footTrack", foot_track, 1e8
                )

        # ---------------------------------------------------------------------------- #
        # Joint states tracking cost ------------------------------------------------- #
        # ---------------------------------------------------------------------------- #
        x_track_weights = np.array(
            [1.0] * 6 + [0.1] * (self.robot_model.nv - 6) + [10] * self.robot_model.nv
        )
        x_track_residual = crocoddyl.ResidualModelState(
            self.state, self.robot_model.defaultState, 0
        )
        x_track_activation = crocoddyl.ActivationModelWeightedQuad(x_track_weights**2)
        x_track_cost = crocoddyl.CostModelResidual(
            self.state, x_track_activation, x_track_residual
        )
        costs.addCost("x_track_cost", x_track_cost, 1e1)

        # ---------------------------------------------------------------------------- #
        # Create impulse action model ------------------------------------------------ #
        # ---------------------------------------------------------------------------- #
        model = crocoddyl.ActionModelImpulseFwdDynamics(
            self.state, impulse_model, costs
        )
        model.JMinvJt_damping = JMinvJt_damping
        model.r_coeff = r_coeff
        return model
