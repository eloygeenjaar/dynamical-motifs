import torch.nn as nn
import numpy as np
import model.modules as modules
from abc import abstractmethod


class BaseModel(nn.Module):
    """
    Base class for all models
    """
    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def reinitialize_subject_weights(self, num_subjects: int):
        print(type(self.in_lin), type(self.out_lin))
        if isinstance(self.in_lin, modules.DecomposedLinear):
            print('Reinitializing in_lin')
            self.in_lin.reinitialize_s(num_subjects)
        if isinstance(self.in_lin._orig_mod, modules.DecomposedLinear):
            print('Reinitializing in_lin')
            self.in_lin._orig_mod.reinitialize_s(num_subjects)
        if isinstance(self.out_lin, modules.DecomposedLinear):
            print('Reinitializing out_lin')
            self.out_lin.reinitialize_s(num_subjects)
