# TODO: Implement the mujoco environment for the Dobot robotic arm
from abc import ABC, abstractmethod
import os
from typing import Any

import numpy as np
import gymnasium as gym
import mujoco as mj
from gymnasium_robotics.envs.robot_env import MujocoRobotEnv
from gymnasium import spaces


def color_range(s: float, v: float, n: int):
    """
    Creates an `n` size discrete color range based on Value and Saturation passed.

    params:
        - s: float  Value in [0, 1]
        - v: float  Saturation in [0, 1]
    """

    s = np.clip(s, 0, 1)
    v = np.clip(v, 0, 1)

    colors = []

    hues = np.linspace(0.0, 1.0, n + 1)
    print(hues)

    for hue in hues[:-1]:

        index = int(hue * 6)
        f = hue * 6 - index
        q = 1 - f

        if index % 6 == 0 or index == 0:
            colors.append([1, f, 0])
        if index == 1:
            colors.append([q, 1, 0])
        if index == 2:
            colors.append([0, 1, f])
        if index == 3:
            colors.append([0, q, 1])
        if index == 4:
            colors.append([f, 0, 1])
        if index == 5:
            colors.append([1, 0, q])

        print(colors[-1])

    return colors


R_MIN = 0.15  # meters
R_MAX = 0.275  # meters
THETA_LIM = np.pi / 3  # rad
TABLE_THICKNESS = 0.02  # meters

PI2_ROT = np.array(
    [
        [np.cos(-np.pi / 2), -np.sin(np.pi / 2)],
        [np.sin(-np.pi / 2), np.cos(np.pi / 2)],
    ]
)

DOBOT_MOTOR_LIMITS = [
    5.59,  # rad/s
    5.59,  # rad/s
    5.59,  # rad/s
    8.38,  # rad/s
    3.0,  # Has no meaning, the rate at which the pump activates ?
]

DOBOT_ACT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_lift",
    "tool_roll",
    "suction_cup_pump",
]

EE_LINK_NAME = "suctionCup_link2"


def bodies_are_colliding(model: mj.MjModel, data: mj.MjData, body1_name, body2_name):
    body1_id = model.body(body1_name).id
    body2_id = model.body(body2_name).id

    # Get all geoms that belong to each body
    geom1_ids = [i for i in range(model.ngeom) if model.geom_bodyid[i] == body1_id]
    geom2_ids = [i for i in range(model.ngeom) if model.geom_bodyid[i] == body2_id]

    for i in range(data.ncon):
        c = data.contact[i]
        if (c.geom1 in geom1_ids and c.geom2 in geom2_ids) or (
            c.geom1 in geom2_ids and c.geom2 in geom1_ids
        ):
            return True
    return False


class DobotBlockEnv(MujocoRobotEnv):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
        ],
        "render_fps": 1000,
    }

    def __init__(
        self,
        n_cubes=2,
        distance_threshold=0.01,
        n_substeps=1,
        default_camera_config=None,
        **kwargs,
    ) -> None:
        self.n_cubes = n_cubes
        self.distance_threshold = distance_threshold
        self.mujoco_renderer = None
        super().__init__(
            default_camera_config=default_camera_config,
            n_substeps=n_substeps,
            model_path=os.path.join(
                os.path.dirname(__file__), "assets", "dobot_table_scene.xml"
            ),
            initial_qpos=0,
            n_actions=5,
            **kwargs,
        )
        self.render()

    def _set_action(self, action) -> None:
        action = np.clip(action, self.action_space.low, self.action_space.high)
        q_target = np.copy(self.data.ctrl)

        delta_q = action * DOBOT_MOTOR_LIMITS * self.dt
        q_target += delta_q

        # update joint targets
        q_target = np.clip(q_target, *(self.jnt_ranges.T))
        if q_target[-1] > 0.5:
            self.suction_activated = True

        self.data.ctrl = q_target

    def _is_success(self, achieved_goal, desired_goal) -> bool:
        return np.linalg.norm(achieved_goal - desired_goal) < self.distance_threshold

    def compute_reward(self, achieved_goal, desired_goal, info) -> float:
        # d_eef_cube = np.linalg.norm()
        d = np.linalg.norm(achieved_goal - desired_goal)
        reward = -d
        if info["is_success"]:
            reward += 5.0
        return reward

    def compute_terminated(self, achieved_goal, desired_goal, info) -> bool:
        return self._is_success(achieved_goal, desired_goal)

    def compute_truncated(self, _achieved_goal, _desired_goal, _info) -> bool:
        return False

    def _step_callback(self) -> None:
        print(f"Suction activated : {self.suction_activated}")
        self.grasped = bodies_are_colliding(
            self.model, self.data, EE_LINK_NAME, "cubeworld_0"
        ) and self.suction_activated

    def _initialize_simulation(self):
        self.spec: mj.MjSpec = mj.MjSpec.from_file(self.fullpath)
        self.jnt_ranges = np.array(
            [self.spec.actuator(name).ctrlrange for name in DOBOT_ACT_NAMES]
        )
        self.act_targets = [self.spec.actuator(name).target for name in DOBOT_ACT_NAMES]
        self.jnt_names = [jnt.name for jnt in self.spec.joints]
        self.grasped = False

        self._randomize_spec()

    def _randomize_spec(self):
        self.randomized_spec = self.spec.copy()

        cube_specs = self._randomize_cube_domain()
        cubes_site = self.randomized_spec.body("dobot").add_site(group=3)

        for i, cube_spec in enumerate(cube_specs):
            cubes_site.attach_body(cube_spec.worldbody, "cube", f"_{i}")

        self.model: mj.MjModel = self.randomized_spec.compile()
        self.data = mj.MjData(self.model)
        mj.mj_forward(self.model, self.data)

        return bodies_are_colliding(self.model, self.data, "cubeworld_0", "cubeworld_1")

    def _reset_sim(self):
        self._randomize_spec()
        self.grasped = False
        self.suction_activated = False

        # Weird hack to update the viewer when randomizing the domain
        # and recompiling the randomized spec
        self.render()
        if self.mujoco_renderer is not None:
            viewer = self.mujoco_renderer._viewers.get(self.render_mode)
            if viewer is not None:
                viewer.model = self.model
                viewer.data = self.data

        return True

    def _sample_goal(self):
        return self.data.body("cubeworld_1").xpos + np.array([0, 0, 0.02])

    def _get_obs(self):
        qpos = [
            self.data.joint(name).qpos[0]
            for name in self.act_targets
            if name in self.jnt_names
        ]
        qdot = [
            self.data.joint(name).qvel[0]
            for name in self.act_targets
            if name in self.jnt_names
        ]
        ee_pos = self.data.body(EE_LINK_NAME).xpos
        ee_quat = self.data.body(EE_LINK_NAME).xquat
        ee_vel = np.zeros(6)
        mj.mj_objectVelocity(
            self.model,
            self.data,
            mj.mjtObj.mjOBJ_BODY,
            self.spec.body(EE_LINK_NAME).id,
            ee_vel,
            0,
        )
        ee_vel = ee_vel[3:]
        cubeA_pos = self.data.body("cubeworld_0").xpos
        cubeA_quat = self.data.body("cubeworld_0").xquat
        cubeB_pos = self.data.body("cubeworld_1").xpos
        cubeB_quat = self.data.body("cubeworld_1").xquat

        return {
            "achieved_goal": cubeA_pos,
            "desired_goal": self.goal,
            "observation": np.concatenate(
                (
                    qpos,
                    qdot,
                    ee_pos,
                    ee_quat,
                    ee_vel,
                    cubeA_pos,
                    cubeA_quat,
                    cubeB_pos,
                    cubeB_quat,
                )
            ),
        }

    def _randomize_cube_domain(self) -> list[mj.MjSpec]:
        """
        Creates a list of `MjSpec` with randomized cubes and returns it
        """
        specs = []
        colors = color_range(0.8, 1.0, self.n_cubes)
        for i in range(self.n_cubes):
            cube_spec = mj.MjSpec()
            cube_spec.compiler.degree = False

            # Sample cube position relative to Dobot in polar coordinates
            r = np.random.uniform(R_MIN, R_MAX)
            theta = np.random.uniform(-THETA_LIM, THETA_LIM)

            x = r * np.cos(theta)
            y = r * np.sin(theta)
            coords = np.array([x, y])
            coords_rot = PI2_ROT @ coords.T

            cube_spec.worldbody.pos = [*coords_rot, TABLE_THICKNESS]
            cube_spec.worldbody.add_geom(
                type=mj.mjtGeom.mjGEOM_BOX,
                size=[0.02, 0.02, 0.02],
                rgba=[*colors[i], 1.0],
                mass=np.random.uniform(0.01, 0.1),
            )

            specs.append(cube_spec)

        return specs


if __name__ == "__main__":
    env = DobotBlockEnv(render_mode="human")

    env.reset(seed=1234)

    i = 0

    while True:
        if i % 1000 == 0:
            env.reset()
        act = np.ones(5)
        env.step(act)
        i += 1
