import numpy as np
import pinocchio
import crocoddyl
from typing import List, Dict
from scipy.spatial.transform import Rotation as R


class BackflipProblem:
    def __init__(
        self,
        robot_model: pinocchio.Model,
        base_frame_name: str,
        rf_contact_frame_name: str,
        lf_contact_frame_name: str,
        integrator: str = "rk4",
        control: str = "zero",
        fwddyn: bool = True,
    ):
        """
        Construct a backflip problem.

        :param robot_model: Pinocchio robot model.
        :param base_frame_name: Name of the base frame.
        :param rf_contact_frame_name: Name of the right foot contact frame.
        :param lf_contact_frame_name: Name of the left foot contact frame.
        :param q_ground: Default joint configuration for the ground phase.
        :param q_flying: Default joint configuration for the flying phase.
        :param q_flying_ready: Default joint configuration for the flying ready phase.
        :param integrator: Type of the integrator ('euler', 'rk4', 'rk3', 'rk2').
        :param control: Type of control parametrization ('zero', 'one', 'rk4', 'rk3').
        :param fwddyn: Use forward dynamics if True, otherwise inverse dynamics.
        """
        self.robot_model = robot_model
        self.robot_data = robot_model.createData()
        self.state = crocoddyl.StateMultibody(self.robot_model)
        self.actuation = crocoddyl.ActuationModelFloatingBase(self.state)
        self._integrator = integrator
        self._control = control
        self._fwddyn = fwddyn

        # Get frame IDs of base and foot contact frames
        self.base_frame_id = self.robot_model.getFrameId(base_frame_name)
        self.lf_contact_frame_id = self.robot_model.getFrameId(lf_contact_frame_name)
        self.rf_contact_frame_id = self.robot_model.getFrameId(rf_contact_frame_name)

        # Default states
        self.x_ground = np.concatenate(
            [
                self.robot_model.referenceConfigurations["half_sitting"].copy(),
                np.zeros(self.robot_model.nv),
            ]
        )
        self.x_flying_ready = np.concatenate(
            [
                self.robot_model.referenceConfigurations["flying_ready"].copy(),
                np.zeros(self.robot_model.nv),
            ]
        )
        self.x_flying_takeoff = np.concatenate(
            [
                self.robot_model.referenceConfigurations["flying_takeoff"].copy(),
                np.zeros(self.robot_model.nv),
            ]
        )
        self.x_flying = np.concatenate(
            [
                self.robot_model.referenceConfigurations["flying"].copy(),
                np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                np.zeros(self.robot_model.nv - 6),
            ]
        )
        self.x_flying_land = np.concatenate(
            [
                self.robot_model.referenceConfigurations["flying_land"].copy(),
                np.zeros(self.robot_model.nv),
            ]
        )

        # Define friction coefficient and ground
        self.mu = 0.7
        self.R_ground = np.eye(3)
        self.x_bounds_weights = np.array(
            [0.0] * 6  # base SE3 residual (no bounds)
            + [100.0] * (self.robot_model.nv - 6)  # joint position residual
            + [0.0] * 3  # base linear velocity residual (no bounds)
            + [100.0] * 3  # base angular velocity residual
            + [100.0] * (self.robot_model.nv - 6)  # joint velocity residual
        )

    def create_backflip_problem_first_stage(
        self,
        x0: np.ndarray,
        jump_height: float,
        jump_length: List[float],
        dt: float,
        num_ground_knots: int,
        num_flying_knots: int,
        v_liftoff: float,
        x_lb: np.ndarray,
        x_ub: np.ndarray,
    ) -> crocoddyl.ShootingProblem:
        """
        Construct a shooting problem for backflip first stage.

        The first stage includes the following phases:
            [Ready, Take-Off, Flying-Up]

        The robot body pitch rotates from 0 to -135 degrees.

        :param x0: Initial state of the robot (q0, v0).
        :param jump_height: Height of the jump in meters.
        :param jump_length: Length of the jump in meters (x, y, z).
        :param dt: Time step length in second.
        :param num_ground_knots: Number of knots on the ground.
        :param num_flying_knots: Number of knots in the air.
        :param v_liftoff: Lift-off velocity in the z direction.
        :param x_lb: Lower bounds for joint states.
        :param x_ub: Upper bounds for joint states.

        :return: A shooting problem for backflip first stage.
        """
        # ---------------------------------------------------------------------------- #
        # Initialize ----------------------------------------------------------------- #
        # ---------------------------------------------------------------------------- #
        # Initialize forward kinematics
        backflip_action_models = []
        q0 = x0[: self.robot_model.nq]
        pinocchio.forwardKinematics(self.robot_model, self.robot_data, q0)
        pinocchio.updateFramePlacements(self.robot_model, self.robot_data)

        # Initial foot contact frames reference SE3
        lf_contact_pose_0 = self.robot_data.oMf[self.lf_contact_frame_id]
        rf_contact_pose_0 = self.robot_data.oMf[self.rf_contact_frame_id]
        foot_poses_ref_0 = {
            self.lf_contact_frame_id: lf_contact_pose_0,
            self.rf_contact_frame_id: rf_contact_pose_0,
        }

        # Initial base reference position
        base_pos_ref_0 = (
            lf_contact_pose_0.translation + rf_contact_pose_0.translation
        ) / 2.0
        base_pos_ref_0[2] = self.robot_data.oMf[self.base_frame_id].translation[2]

        # ---------------------------------------------------------------------------- #
        # Ready Phase ---------------------------------------------------------------- #
        # ---------------------------------------------------------------------------- #
        # Control the robot to track the default flying-ready joint configuration
        x_track_weights_ready = np.array(
            [0.0, 100.0, 0.0]  # x, y, z
            + [100.0, 0.0, 100.0]  # axis-angle residual rotation s.t. R = R_ref * r
            + [50.0] * (self.state.nv - 6)  # joint positions
            + [25.0, 100.0, 0.0]  # base linear velocities
            + [100.0, 0.0, 100.0]  # base angular velocities
            + [1.0] * (self.state.nv - 6)  # joint velocities
        )
        x_ref_traj_ready = self.state_interp(x0, self.x_flying_ready, num_ground_knots)
        ready = [
            self.create_knot_action_model(
                dt,
                [self.lf_contact_frame_id, self.rf_contact_frame_id],
                foot_poses_ref=foot_poses_ref_0,
                x_ref=x_ref_traj_ready[k],
                x_track_weights=x_track_weights_ready,
                x_track_cost_weight=1e3,
                x_lb=x_lb,
                x_ub=x_ub,
                x_bounds_cost_weight=1e12,
            )
            for k in range(num_ground_knots)
        ]
        ready[-1] = self.create_knot_action_model(
            dt,
            [self.lf_contact_frame_id, self.rf_contact_frame_id],
            foot_poses_ref=foot_poses_ref_0,
            x_ref=self.x_flying_ready,
            x_track_weights=np.array(
                [25.0, 100.0, 0.0]  # x, y, z
                + [100.0, 0.0, 100.0]  # axis-angle residual rotation s.t. R = R_ref * r
                + [75.0] * (self.state.nv - 6)  # joint positions
                + [0.0, 100.0, 0.0]  # base linear velocities
                + [100.0, 0.0, 100.0]  # base angular velocities
                + [0.0] * (self.state.nv - 6)  # joint velocities
            ),
            x_track_cost_weight=1e6,
            x_lb=x_lb,
            x_ub=x_ub,
            x_bounds_cost_weight=1e12,
        )

        # ---------------------------------------------------------------------------- #
        # Take-Off Phase ------------------------------------------------------------- #
        # ---------------------------------------------------------------------------- #
        # Control the robot to track the default flying-ready joint configuration
        x_track_weights_takeoff = np.array(
            [25.0, 100.0, 0.0]  # x, y, z
            + [100.0, 0.0, 100.0]  # axis-angle residual rotation s.t. R = R_ref * r
            + [50.0] * (self.state.nv - 6)  # joint positions
            + [25.0, 100.0, 10.0]  # base linear velocities
            + [100.0, 0.0, 100.0]  # base angular velocities
            + [0.0] * (self.state.nv - 6)  # joint velocities
        )
        self.x_flying_takeoff[self.robot_model.nq : self.robot_model.nq + 3] = np.array(
            [0.0, 0.0, v_liftoff]
        )
        x_ref_traj_takeoff = self.state_interp(
            self.x_flying_ready, self.x_flying_takeoff, num_ground_knots
        )
        take_off = [
            self.create_knot_action_model(
                dt,
                [self.lf_contact_frame_id, self.rf_contact_frame_id],
                foot_poses_ref=foot_poses_ref_0,
                x_ref=x_ref_traj_takeoff[k],
                x_track_weights=x_track_weights_takeoff,
                x_track_cost_weight=1e3,
                x_lb=x_lb,
                x_ub=x_ub,
                x_bounds_cost_weight=1e12,
            )
            for k in range(num_ground_knots)
        ]
        take_off[-1] = self.create_knot_action_model(
            dt,
            [self.lf_contact_frame_id, self.rf_contact_frame_id],
            foot_poses_ref=foot_poses_ref_0,
            x_ref=self.x_flying_takeoff,
            x_track_weights=np.array(
                [25.0, 100.0, 100.0]  # x, y, z
                + [100.0, 0.0, 100.0]  # axis-angle residual rotation s.t. R = R_ref * r
                + [50.0] * (self.state.nv - 6)  # joint positions
                + [0.0, 0.0, 200.0]  # base linear velocities
                + [100.0, 0.0, 100.0]  # base angular velocities
                + [0.0] * (self.state.nv - 6)  # joint velocities
            ),
            x_track_cost_weight=1e6,
            x_lb=x_lb,
            x_ub=x_ub,
            x_bounds_cost_weight=1e12,
        )

        # ---------------------------------------------------------------------------- #
        # Flying-Up Phase ------------------------------------------------------------ #
        # ---------------------------------------------------------------------------- #
        # Control the robot to track a whole-base state reference trajectory
        base_pos_ref_0 = self.x_flying_takeoff[:3]
        base_pos_ref_peak = base_pos_ref_0 + np.array(
            [
                jump_length[0] / 2.0,
                jump_length[1] / 2.0,
                jump_length[2] / 2.0 + jump_height,
            ]
        )
        base_pos_ref_traj = [
            base_pos_ref_0
            + (base_pos_ref_peak - base_pos_ref_0) * (k + 1) / num_flying_knots
            for k in range(num_flying_knots)
        ]
        base_pitch_ref_traj = [
            -(3 * np.pi / 4) * (k + 1) / num_flying_knots
            for k in range(num_flying_knots)
        ]
        base_quat_ref_traj = [
            R.from_euler("zyx", [0.0, base_pitch_ref_traj[k], 0.0]).as_quat()
            for k in range(num_flying_knots)
        ]
        base_pitch_ang_vel_ref = -(3 * np.pi / 4) / (num_flying_knots * dt)
        base_ang_vel_ref = [
            np.array([0.0, base_pitch_ang_vel_ref, 0.0])
            for k in range(num_flying_knots)
        ]
        x_ref_traj_flyup = np.vstack(
            [
                self.state_interp(
                    self.x_flying_takeoff, self.x_flying, num_flying_knots - 10
                ),
                np.tile(self.x_flying, (10, 1)),
            ]
        )
        x_ref_traj_flyup[:, :3] = np.array(base_pos_ref_traj)
        x_ref_traj_flyup[:, 3:7] = np.array(base_quat_ref_traj)
        x_ref_traj_flyup[:, self.robot_model.nq + 3 : self.robot_model.nq + 6] = (
            base_ang_vel_ref
        )

        # Create knot action models
        x_track_weights_flyup = np.array(
            [0.0, 100.0, 0.0]  # x, y, z
            + [100.0, 10.0, 100.0]  # axis-angle residual rotation s.t. R = R_ref * r
            + [75.0] * (self.state.nv - 6)  # joint positions
            + [0.0, 100.0, 0.0]  # base linear velocities
            + [100.0, 100.0, 100.0]  # base angular velocities
            + [0.5] * (self.state.nv - 6)  # joint velocities
        )
        fly_up = [
            self.create_knot_action_model(
                dt,
                [],
                foot_poses_ref=None,
                x_ref=x_ref_traj_flyup[k],
                x_track_weights=x_track_weights_flyup,
                x_track_cost_weight=1e3,
                x_lb=x_lb,
                x_ub=x_ub,
                x_bounds_cost_weight=1e12,
            )
            for k in range(num_flying_knots)
        ]
        fly_up[-1] = self.create_knot_action_model(
            dt,
            [],
            foot_poses_ref=None,
            x_ref=x_ref_traj_flyup[-1],
            x_track_weights=np.array(
                [0.0, 100.0, 50.0]  # x, y, z
                + [100.0] * 3  # axis-angle residual rotation s.t. R = R_ref * r
                + [100.0] * (self.state.nv - 6)  # joint positions
                + [0.0, 100.0, 50.0]  # base linear velocities
                + [100.0, 100.0, 100.0]  # base angular velocities
                + [0.0] * (self.state.nv - 6)  # joint velocities
            ),
            x_track_cost_weight=1e6,
            x_lb=x_lb,
            x_ub=x_ub,
            x_bounds_cost_weight=1e12,
        )

        # ---------------------------------------------------------------------------- #
        # Create Shooting Problem ---------------------------------------------------- #
        # ---------------------------------------------------------------------------- #
        backflip_action_models += ready
        backflip_action_models += take_off
        backflip_action_models += fly_up
        return crocoddyl.ShootingProblem(
            x0, backflip_action_models[:-1], backflip_action_models[-1]
        )

    def create_backflip_problem_second_stage(
        self,
        x0: np.ndarray,
        jump_length: List[float],
        dt: float,
        num_ground_knots: int,
        num_flying_knots: int,
        x_lb: np.ndarray,
        x_ub: np.ndarray,
    ) -> crocoddyl.ShootingProblem:
        """
        Construct a shooting problem for backflip second stage.

        The second stage includes the following phases:
            [Flying-Down, Landing, Landed]

        The robot body pitch rotates from -135 to -360 degrees.

        :param x0: Initial state of the robot (q0, v0).
        :param jump_length: Length of the jump in meters (x, y, z).
        :param dt: Time step length in second.
        :param num_ground_knots: Number of knots on the ground.
        :param num_flying_knots: Number of knots in the air.
        :param x_lb: Lower bounds for joint states.
        :param x_ub: Upper bounds for joint states.

        :return: A shooting problem for backflip second stage.
        """
        backflip_action_models = []
        pinocchio.forwardKinematics(
            self.robot_model, self.robot_data, self.x_ground[: self.robot_model.nq]
        )
        pinocchio.updateFramePlacements(self.robot_model, self.robot_data)

        # ---------------------------------------------------------------------------- #
        # Flying-Down Phase ---------------------------------------------------------- #
        # ---------------------------------------------------------------------------- #
        # Control the robot to track a whole-base state reference trajectory
        foot_poses_ref_final = {
            self.lf_contact_frame_id: pinocchio.SE3(
                np.eye(3), np.array([0.48, 0.09, 0.0])
            ),
            self.rf_contact_frame_id: pinocchio.SE3(
                np.eye(3), np.array([0.48, -0.09, 0.0])
            ),
        }

        # Update base position reference
        self.x_flying_land[:3] += np.array(jump_length)
        base_pitch_0 = R.from_quat(x0[3:7]).as_euler("yzx")[0]
        base_pitch_f = R.from_quat(self.x_flying_land[3:7]).as_euler("yzx")[0]
        base_pitch_ref_traj = [
            base_pitch_0 - (base_pitch_f - base_pitch_0) * (k + 1) / num_flying_knots
            for k in range(num_flying_knots)
        ]
        base_quat_ref_traj = [
            R.from_euler("zyx", [0.0, base_pitch_ref_traj[k], 0.0]).as_quat()
            for k in range(num_flying_knots)
        ]
        base_pitch_ang_vel_ref = (base_pitch_f - base_pitch_0) / (num_flying_knots * dt)

        x_ref_traj_flydown = self.state_interp(x0, self.x_flying_land, num_flying_knots)
        x_ref_traj_flydown[:, 3:7] = np.array(base_quat_ref_traj)
        x_ref_traj_flydown[:, self.robot_model.nq : self.robot_model.nq + 3] = np.tile(
            np.zeros(3), (num_flying_knots, 1)
        )
        x_ref_traj_flydown[:, self.robot_model.nq + 3] = np.zeros(num_flying_knots)
        x_ref_traj_flydown[:, self.robot_model.nq + 4] = (
            base_pitch_ang_vel_ref * np.ones(num_flying_knots)
        )
        x_ref_traj_flydown[:, self.robot_model.nq + 5] = np.zeros(num_flying_knots)

        joint_pos_0 = x0[7 : self.robot_model.nq]
        joint_pos_ref_land = self.x_flying_land[7 : self.robot_model.nq]
        joint_pos_ref_traj = [
            joint_pos_0
            + (joint_pos_ref_land - joint_pos_0) * (k + 1) / (num_flying_knots - 12)
            for k in range(num_flying_knots - 12)
        ] + [joint_pos_ref_land] * 12
        x_ref_traj_flydown[:, 7 : self.robot_model.nq] = np.array(joint_pos_ref_traj)

        # Create knot action models
        x_track_weights_flydown = np.array(
            [0.0, 0.0, 0.0]  # x, y, z
            + [0.0, 0.0, 0.0]  # axis-angle residual rotation s.t. R = R_ref * r
            + [100.0] * (self.state.nv - 6)  # joint positions
            + [0.0, 0.0, 0.0]  # base linear velocities
            + [0.0, 0.0, 0.0]  # base angular velocities
            + [0.1] * (self.state.nv - 6)  # joint velocities
        )
        fly_down = [
            self.create_knot_action_model(
                dt,
                [],
                foot_poses_ref=None,
                x_ref=x_ref_traj_flydown[k],
                x_track_weights=x_track_weights_flydown,
                x_track_cost_weight=1e5,
                x_lb=x_lb,
                x_ub=x_ub,
                x_bounds_cost_weight=1e12,
            )
            for k in range(num_flying_knots)
        ]
        fly_down[-1] = self.create_knot_action_model(
            dt,
            [],
            foot_poses_ref=foot_poses_ref_final,
            track_foot_poses=True,
            foot_poses_track_cost_weight=1e8,
            x_ref=x_ref_traj_flydown[-1],
            x_track_weights=np.array(
                [0.0, 0.0, 0.0]  # x, y, z
                + [0.0] * 3  # axis-angle residual rotation s.t. R = R_ref * r
                + [100.0] * (self.state.nv - 6)  # joint positions
                + [0.0, 0.0, 0.0]  # base linear velocities
                + [0.0, 0.0, 0.0]  # base angular velocities
                + [0.0] * (self.state.nv - 6)  # joint velocities
            ),
            x_track_cost_weight=1e6,
            x_lb=x_lb,
            x_ub=x_ub,
            x_bounds_cost_weight=1e12,
        )

        # ---------------------------------------------------------------------------- #
        # Landing Phase -------------------------------------------------------------- #
        # ---------------------------------------------------------------------------- #
        # Create a mode jump map and track the reference feet poses
        landing = [
            self.create_knot_impulse_model(
                [self.lf_contact_frame_id, self.rf_contact_frame_id],
                foot_poses_ref=foot_poses_ref_final,
                foot_poses_track_cost_weight=1e6,
                x_ref=x_ref_traj_flydown[-1],
                x_track_weights=np.array(
                    [0.0, 0.0, 0.0]  # x, y, z
                    + [0.0] * 3  # axis-angle residual rotation s.t. R = R_ref * r
                    + [100.0] * (self.state.nv - 6)  # joint positions
                    + [0.0, 0.0, 0.0]  # base linear velocities
                    + [0.0, 0.0, 0.0]  # base angular velocities
                    + [0.0] * (self.state.nv - 6)  # joint velocities
                ),
                x_track_cost_weight=1e3,
            )
        ]

        # During the landed phase, control the robot to track the final base pose reference
        x_ref_final = np.concatenate([np.array([0.48, 0.0]), self.x_ground[2:]])
        landed = [
            self.create_knot_action_model(
                dt,
                [self.lf_contact_frame_id, self.rf_contact_frame_id],
                foot_poses_ref=foot_poses_ref_final,
                x_ref=x_ref_final,
                x_track_weights=np.array(
                    [0.0, 0.0, 0.0]  # x, y, z
                    + [0.0] * 3  # axis-angle residual rotation s.t. R = R_ref * r
                    + [100.0] * (self.state.nv - 6)  # joint positions
                    + [0.0, 0.0, 0.0]  # base linear velocities
                    + [0.0, 0.0, 0.0]  # base angular velocities
                    + [5.0] * (self.state.nv - 6)  # joint velocities
                ),
                x_track_cost_weight=1e3,
            )
            for _ in range(num_ground_knots)
        ]

        # ---------------------------------------------------------------------------- #
        # Create Shooting Problem ---------------------------------------------------- #
        # ---------------------------------------------------------------------------- #
        backflip_action_models += fly_down
        backflip_action_models += landing
        backflip_action_models += landed
        return crocoddyl.ShootingProblem(
            x0, backflip_action_models[:-1], backflip_action_models[-1]
        )

    def create_knot_action_model(
        self,
        dt: float,
        support_foot_ids: List[int],
        wrench_cone_cost_weight: float = 1e1,
        foot_poses_ref: Dict[int, pinocchio.SE3] = None,
        track_foot_poses: bool = False,
        foot_poses_track_cost_weight: float = 1e6,
        x_ref: np.ndarray = None,
        x_track_weights: np.ndarray = None,
        x_track_cost_weight: float = None,
        x_lb: np.ndarray = None,
        x_ub: np.ndarray = None,
        x_bounds_cost_weight: float = 1e6,
        ctrl_regu_cost_weight: float = 1e-8,
    ) -> crocoddyl.IntegratedActionModelAbstract:
        """
        Create an action model for one knot.

        :param dt: time step length in second.
        :param support_foot_ids: IDs of support feet.
        :param wrench_cone_cost_weight: Weight for the wrench cone cost.
        :param foot_poses_ref: Dictionary of swing feet SE3 references.
        :param track_foot_poses: Whether to track the foot poses.
        :param foot_poses_track_cost_weight: Weight for the foot poses tracking cost.
        :param x_ref: State reference (q, v).
        :param x_track_weights: Weight vector in the state tracking cost.
        :param x_track_cost_weight: Weight for the state tracking cost.
        :param x_lb: Lower bounds for joint states.
        :param x_ub: Upper bounds for joint states.
        :param x_bounds_cost_weight: Weight for the joint states bounds cost.
        :param ctrl_regu_cost_weight: Weight for the control regularization cost.

        :return: An action model for one knot.
        """
        if len(support_foot_ids) > 0:
            if foot_poses_ref is None:
                raise ValueError("foot_poses_ref must be provided.")
        if track_foot_poses:
            if foot_poses_ref is None:
                raise ValueError("foot_poses_ref must be provided.")
            if foot_poses_track_cost_weight is None:
                raise ValueError("foot_poses_track_cost_weight must be provided.")
        if x_ref is not None:
            if x_track_weights is None:
                raise ValueError("x_track_weights must be provided.")
            if x_track_cost_weight is None:
                raise ValueError("x_track_cost_weight must be provided.")

        nu = self.actuation.nu
        costs = crocoddyl.CostModelSum(self.state, nu)

        # ---------------------------------------------------------------------------- #
        # Contact models for support feet -------------------------------------------- #
        # ---------------------------------------------------------------------------- #
        contact_models = crocoddyl.ContactModelMultiple(self.state, nu)
        for foot_id in support_foot_ids:
            foot_contact_model = crocoddyl.ContactModel6D(
                self.state,
                foot_id,
                foot_poses_ref[foot_id],
                pinocchio.LOCAL_WORLD_ALIGNED,
                nu,
                np.array([0.0, 30.0]),
            )
            contact_models.addContact(
                self.robot_model.frames[foot_id].name + "_contact", foot_contact_model
            )

        # ---------------------------------------------------------------------------- #
        # Contact wrench cone costs -------------------------------------------------- #
        # ---------------------------------------------------------------------------- #
        foot_size = np.array([0.1, 0.05])  # [length, width]
        for foot_id in support_foot_ids:
            cone = crocoddyl.WrenchCone(self.R_ground, self.mu, foot_size)
            wrench_residual = crocoddyl.ResidualModelContactWrenchCone(
                self.state, foot_id, cone, nu, fwddyn=True
            )
            wrench_activation = crocoddyl.ActivationModelQuadraticBarrier(
                crocoddyl.ActivationBounds(cone.lb, cone.ub)
            )
            wrench_cone_cost = crocoddyl.CostModelResidual(
                self.state, wrench_activation, wrench_residual
            )
            costs.addCost(
                self.robot_model.frames[foot_id].name + "_wrench_cone",
                wrench_cone_cost,
                wrench_cone_cost_weight,
            )

        # ---------------------------------------------------------------------------- #
        # Foot poses tracking costs -------------------------------------------------- #
        # ---------------------------------------------------------------------------- #
        if track_foot_poses:
            for foot_id, foot_poses_ref in foot_poses_ref.items():
                foot_poses_track_residual = crocoddyl.ResidualModelFramePlacement(
                    self.state, foot_id, foot_poses_ref, nu
                )
                foot_poses_track_cost = crocoddyl.CostModelResidual(
                    self.state, foot_poses_track_residual
                )
                costs.addCost(
                    self.robot_model.frames[foot_id].name + "_foot_poses_track",
                    foot_poses_track_cost,
                    foot_poses_track_cost_weight,
                )

        # ---------------------------------------------------------------------------- #
        # Whole-base joint state tracking cost --------------------------------------- #
        # ---------------------------------------------------------------------------- #
        if x_ref is not None:
            x_track_residual = crocoddyl.ResidualModelState(self.state, x_ref, nu)
            x_track_activation = crocoddyl.ActivationModelWeightedQuad(
                x_track_weights**2
            )
            x_track_cost = crocoddyl.CostModelResidual(
                self.state, x_track_activation, x_track_residual
            )
            costs.addCost("x_track", x_track_cost, x_track_cost_weight)

        # ---------------------------------------------------------------------------- #
        # Whole-base joint state bounds ---------------------------------------------- #
        # ---------------------------------------------------------------------------- #
        if x_lb is not None and x_ub is not None:
            x_bounds_residual = crocoddyl.ResidualModelState(self.state, x_lb, nu)

            # Construct bounds for residuals
            r_lb = np.zeros(2 * self.robot_model.nv)
            r_ub = self.state.diff(x_lb, x_ub)
            r_ub[3:6] = 1e6 * np.ones(3)  # base orientation residuals

            # Create cost
            x_bounds_activation = crocoddyl.ActivationModelWeightedQuadraticBarrier(
                crocoddyl.ActivationBounds(r_lb, r_ub),
                self.x_bounds_weights,
            )
            x_bounds_cost = crocoddyl.CostModelResidual(
                self.state, x_bounds_activation, x_bounds_residual
            )
            costs.addCost("x_bounds", x_bounds_cost, x_bounds_cost_weight)

        # ---------------------------------------------------------------------------- #
        # Control regularization cost ------------------------------------------------ #
        # ---------------------------------------------------------------------------- #
        ctrl_residual = crocoddyl.ResidualModelControl(self.state, nu)
        ctrl_regu_cost = crocoddyl.CostModelResidual(self.state, ctrl_residual)
        costs.addCost("control_regularization", ctrl_regu_cost, ctrl_regu_cost_weight)

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

    def create_knot_impulse_model(
        self,
        support_foot_ids: List[int],
        foot_poses_ref: Dict[int, pinocchio.SE3],
        x_ref: np.ndarray = None,
        foot_poses_track_cost_weight: float = 1e6,
        x_track_weights: np.ndarray = None,
        x_track_cost_weight: float = 1e3,
        r_coeff: float = 0.0,
        JMinvJt_damping: float = 1e-12,
    ) -> crocoddyl.ActionModelImpulseFwdDynamics:
        """
        Create an actional model for impulse dynamics. This is used to handle the mode
        switch caused by contacts.

        :param support_foot_ids: IDs of support feet.
        :param foot_poses_ref: Dictionary of swing feet SE3 references.
        :param x_ref: State reference (q, v).
        :param foot_poses_track_cost_weight: Weight for the foot poses tracking cost.
        :param x_track_weights: Weight vector in the state tracking cost.
        :param x_track_cost_weight: Weight for the state tracking cost.
        :param r_coeff: Restitution coefficient.
        :param JMinvJt_damping: Damping term used in operational space inertia matrix.

        :return: An impulse action model.
        """
        if x_ref is not None:
            if x_track_weights is None:
                raise ValueError("x_track_weights must be provided.")

        # ---------------------------------------------------------------------------- #
        # Impulse models for support feet -------------------------------------------- #
        # ---------------------------------------------------------------------------- #
        impulse_models = crocoddyl.ImpulseModelMultiple(self.state)
        for foot_id in support_foot_ids:
            foot_impulse_model = crocoddyl.ImpulseModel6D(
                self.state, foot_id, pinocchio.LOCAL_WORLD_ALIGNED
            )
            impulse_models.addImpulse(
                self.robot_model.frames[foot_id].name + "_impulse", foot_impulse_model
            )

        # ---------------------------------------------------------------------------- #
        # Foot poses tracking costs -------------------------------------------------- #
        # ---------------------------------------------------------------------------- #
        nu = 0  # no actuation during impulse
        costs = crocoddyl.CostModelSum(self.state, nu)
        for foot_id, foot_poses_ref in foot_poses_ref.items():
            foot_poses_track_residual = crocoddyl.ResidualModelFramePlacement(
                self.state, foot_id, foot_poses_ref, nu
            )
            foot_poses_track_cost = crocoddyl.CostModelResidual(
                self.state, foot_poses_track_residual
            )
            costs.addCost(
                self.robot_model.frames[foot_id].name + "_foot_poses_track",
                foot_poses_track_cost,
                foot_poses_track_cost_weight,
            )

        # ---------------------------------------------------------------------------- #
        # Joint states tracking cost ------------------------------------------------- #
        # ---------------------------------------------------------------------------- #
        x_track_residual = crocoddyl.ResidualModelState(self.state, x_ref, 0)
        x_track_activation = crocoddyl.ActivationModelWeightedQuad(x_track_weights**2)
        x_track_cost = crocoddyl.CostModelResidual(
            self.state, x_track_activation, x_track_residual
        )
        costs.addCost("x_track", x_track_cost, x_track_cost_weight)

        # ---------------------------------------------------------------------------- #
        # Create impulse action model ------------------------------------------------ #
        # ---------------------------------------------------------------------------- #
        impulse_model = crocoddyl.ActionModelImpulseFwdDynamics(
            self.state, impulse_models, costs, r_coeff, JMinvJt_damping
        )
        return impulse_model

    def state_interp(
        self, x0: np.ndarray, xf: np.ndarray, N: int, type: str = "linear"
    ) -> List[np.ndarray]:
        """
        Generate a list of interpolated states between x0 and x1.

        :param x0: Initial state.
        :param xf: Final state.
        :param N: Length of the resulting trajectory.
        :param type: Type of interpolation.

        :return: A list of interpolated states.
        """
        q0 = x0[: self.robot_model.nq]
        v0 = x0[self.robot_model.nq :]
        base_pos_0 = q0[:3]
        base_quat_0 = q0[3:7]
        base_eul_0 = R.from_quat(base_quat_0).as_euler("zyx")
        joint_pos_0 = q0[7:]
        base_vel_0 = v0[:6]
        joint_vel_0 = v0[6:]

        qf = xf[: self.robot_model.nq]
        vf = xf[self.robot_model.nq :]
        base_pos_f = qf[:3]
        base_quat_f = qf[3:7]
        base_eul_f = R.from_quat(base_quat_f).as_euler("zyx")
        joint_pos_f = qf[7:]
        base_vel_f = vf[:6]
        joint_vel_f = vf[6:]

        x_traj = np.zeros((N, self.robot_model.nq + self.robot_model.nv))
        if type == "linear":
            base_pos_traj = [
                base_pos_0 + (base_pos_f - base_pos_0) * k / (N - 1) for k in range(N)
            ]
            base_eul_traj = [
                base_eul_0 + (base_eul_f - base_eul_0) * k / (N - 1) for k in range(N)
            ]
            base_quat_traj = [
                R.from_euler("zyx", base_eul_traj[k]).as_quat() for k in range(N)
            ]
            joint_pos_traj = [
                joint_pos_0 + (joint_pos_f - joint_pos_0) * k / (N - 1)
                for k in range(N)
            ]
            base_vel_traj = [
                base_vel_0 + (base_vel_f - base_vel_0) * k / (N - 1) for k in range(N)
            ]
            joint_vel_traj = [
                joint_vel_0 + (joint_vel_f - joint_vel_0) * k / (N - 1)
                for k in range(N)
            ]
            x_traj = np.hstack(
                [
                    np.array(base_pos_traj),
                    np.array(base_quat_traj),
                    np.array(joint_pos_traj),
                    np.array(base_vel_traj),
                    np.array(joint_vel_traj),
                ]
            )
        else:
            raise ValueError(f"Interpolation type {type} is not supported.")

        return x_traj
