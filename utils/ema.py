from __future__ import division
from __future__ import unicode_literals

import torch


def get_param_buffer_for_ema(model, update_buffer=False,
                             required_buffers=['running_mean', 'running_var']):
    result = {}

    for name, param in model.named_parameters():
        if param.requires_grad:
            result[name] = param

    if update_buffer:
        for name, buf in model.named_buffers():
            if any(b in name for b in required_buffers):
                result[name] = buf

    return result


class ExponentialMovingAverage:
    """
    Maintains (exponential) moving average of a set of parameters.
    """
    def __init__(self, parameters, decay, use_num_updates=True):
        """
        Args:
          parameters: Iterable of `torch.nn.Parameter`; usually the result of
            `model.parameters()`.
          decay: The exponential decay.
          use_num_updates: Whether to use number of updates when computing
            averages.
        """
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')

        self.decay = decay
        self.num_updates = 0 if use_num_updates else None

        # name -> tensor
        self.shadow_params = {
            name: param.detach().clone()
            for name, param in parameters.items()
        }

        self.collected_params = {}

    def update(self, parameters):
        """
        Update currently maintained parameters.
        Call this every time the parameters are updated, such as the result of
        the `optimizer.step()` call.
        Args:
            parameters: Iterable of `torch.nn.Parameter`; usually the same set of
            parameters used to initialize this object.
        """
        decay = self.decay
        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(decay,
                        (1 + self.num_updates) / (10 + self.num_updates))

        one_minus_decay = 1.0 - decay

        with torch.no_grad():
            for name, param in parameters.items():
                if name not in self.shadow_params:
                    continue

                shadow_param = self.shadow_params[name]

                # keep device consistent
                if shadow_param.device != param.device:
                    shadow_param = shadow_param.to(param.device)
                    self.shadow_params[name] = shadow_param

                shadow_param.sub_(one_minus_decay * (shadow_param - param))

    def copy_to(self, parameters):
        """
        Copy current parameters into given collection of parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored moving averages.
        """
        for name, param in parameters.items():
            if name in self.shadow_params:
                param.data.copy_(self.shadow_params[name].data)

    def store(self, parameters):
        """
        Save the current parameters for restoring later.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            temporarily stored.
        """
        self.collected_params = {
            name: param.detach().clone()
            for name, param in parameters.items()
        }

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored parameters.
        """
        for name, param in parameters.items():
            if name in self.collected_params:
                param.data.copy_(self.collected_params[name].data)

        self.collected_params = {}