# Copyright (C) 2021 THL A29 Limited, a Tencent company.
# All rights reserved.
# Licensed under the BSD 3-Clause License (the "License"); you may
# not use this file except in compliance with the License. You may
# obtain a copy of the License at
# https://opensource.org/licenses/BSD-3-Clause
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" basis,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.
# See the AUTHORS file for names of contributors.

from typing import List

import torch


def lamb(
    params: List[torch.Tensor],
    grads: List[torch.Tensor],
    exp_avgs: List[torch.Tensor],
    exp_avg_sqs: List[torch.Tensor],
    max_exp_avg_sqs: List[torch.Tensor],
    state_steps: List[int],
    *,
    amsgrad: bool,
    beta1: float,
    beta2: float,
    lr: float,
    weight_decay: float,
    eps: float,
    maximize: bool,
    adam: bool,
):
    """lamb implementation extracted from cybertronai/pytorch-lamb
    https://github.com/cybertronai/pytorch-lamb/blob/master/pytorch_lamb/lamb.py
    """
    for i, param in enumerate(params):

        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]

        # Decay the first and second moment running average coefficient
        # m_t
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        # v_t
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        # Paper v3 does not use debiasing.
        # bias_correction1 = 1 - beta1 ** state['step']
        # bias_correction2 = 1 - beta2 ** state['step']
        # Apply bias to lr to avoid broadcast.
        step_size = lr  # * math.sqrt(bias_correction2) / bias_correction1

        weight_norm = param.data.pow(2).sum().sqrt().clamp(0, 10)

        adam_step = exp_avg / exp_avg_sq.sqrt().add(eps)
        if weight_decay != 0:
            adam_step.add_(param.data, alpha=weight_decay)

        adam_norm = adam_step.pow(2).sum().sqrt()
        if weight_norm == 0 or adam_norm == 0:
            trust_ratio = 1
        else:
            trust_ratio = weight_norm / adam_norm
        if adam:
            trust_ratio = 1

        param.data.add_(adam_step, alpha=-step_size * trust_ratio)
