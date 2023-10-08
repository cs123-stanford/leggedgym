# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from time import time
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
# from torch.tensor import Tensor
from typing import Tuple, Dict

from legged_gym.envs import LeggedRobot
from legged_gym import LEGGED_GYM_ROOT_DIR
from .pupper_config import PupperFlatCfg

# NUM_ACTUATORS = 4
# NUM_PHYSICAL_JOINTS = 6

# MIN_LEG_POS = -0.35
# MAX_LEG_POS = -0.15
# MAX_WHEEL_VEL = 30.0
# ACTION_MIN = [MIN_LEG_POS, -MAX_WHEEL_VEL, MIN_LEG_POS, -MAX_WHEEL_VEL]
# ACTION_MAX = [MAX_LEG_POS, MAX_WHEEL_VEL, MAX_LEG_POS, MAX_WHEEL_VEL]

# DEFAULT_DOF_POS = torch.tensor([-0.25, 0.0, -0.25, 0.0], dtype=torch.float, device="cuda", requires_grad=False)

# ACTION_MIN = torch.tensor(ACTION_MIN, dtype=torch.float, device="cuda", requires_grad=False)
# ACTION_MAX = torch.tensor(ACTION_MAX, dtype=torch.float, device="cuda", requires_grad=False)


RESET_PROJECTED_GRAVITY_Z = 0#np.cos(1.04) # Pitch/roll angle to trigger resets

# PITCH_OFFSET_RANGE = [0.0, 0.0] #[-0.05, 0.05]

class Pupper(LeggedRobot):
    cfg : PupperFlatCfg
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    def _process_rigid_body_props(self, props, env_id):
            if self.cfg.domain_rand.randomize_base_mass:
                rng_mass = self.cfg.domain_rand.added_mass_range
                rand_mass = np.random.uniform(rng_mass[0], rng_mass[1], size=(1, ))
                props[0].mass += rand_mass
            else:
                rand_mass = np.zeros(1)
            if self.cfg.domain_rand.randomize_base_com:
                rng_com_x = self.cfg.domain_rand.added_com_range_x
                rng_com_y = self.cfg.domain_rand.added_com_range_y
                rng_com_z = self.cfg.domain_rand.added_com_range_z
                rand_com = np.random.uniform([rng_com_x[0], rng_com_y[0], rng_com_z[0]], [rng_com_x[1], rng_com_y[1], rng_com_z[1]], size=(3, ))
                props[0].com += gymapi.Vec3(*rand_com)
            else:
                rand_com = np.zeros(3)
            mass_params = np.concatenate([rand_mass, rand_com])
            return props

    # TODO: WRITE YOUR CODE HERE
    def _reward_base_height(self):
        # Penalize base height away from target, make sure this value is positive
        return 0.0
    # TODO: WRITE YOUR CODE HERE


    # TODO: WRITE YOUR CODE HERE
    def _reward_forward_velocity(self):
            return 0
    # TODO: WRITE YOUR CODE HERE

    # TODO: WRITE YOUR CODE HERE
    def _reward_torques(self):
            return 0
    # TODO: WRITE YOUR CODE HERE