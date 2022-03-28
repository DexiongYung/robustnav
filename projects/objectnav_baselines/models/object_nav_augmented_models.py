from typing import Tuple, Dict, Optional, cast

import gym
import torch
import torch.nn as nn
from gym.spaces.dict import Dict as SpaceDict

from allenact.algorithms.onpolicy_sync.policy import (
    ActorCriticModel,
    LinearCriticHead,
    LinearActorHead,
    DistributionType,
    Memory,
    ObservationType,
)
from allenact.base_abstractions.distributions import CategoricalDistr
from allenact.base_abstractions.misc import ActorCriticOutput
from allenact.base_abstractions.preprocessor import ResNetEmbedder
from allenact.embodiedai.preprocessors.resnet import ResNetPreprocessor
from allenact.embodiedai.models.basic_models import SimpleCNN, RNNStateEncoder
from .object_nav_models import ResnetDualTensorGoalEncoder, ResnetTensorGoalEncoder

class AugmentedResnetTensorObjectNavActorCritic(ActorCriticModel[CategoricalDistr]):
    def __init__(
        self,
        action_space: gym.spaces.Discrete,
        observation_space: SpaceDict,
        goal_sensor_uuid: str,
        rgb_resnet_preprocessor_uuid: Optional[str],
        resnet_model,
        pool: bool,
        depth_resnet_preprocessor_uuid: Optional[str] = None,
        hidden_size: int = 512,
        goal_dims: int = 32,
        resnet_compressor_hidden_out_dims: Tuple[int, int] = (128, 32),
        combiner_hidden_out_dims: Tuple[int, int] = (128, 32),
        include_auxiliary_head: bool = False,
    ):

        super().__init__(
            action_space=action_space, observation_space=observation_space,
        )

        self._hidden_size = hidden_size
        self.rgb_resnet_preprocessor_uuid = rgb_resnet_preprocessor_uuid
        self.include_auxiliary_head = include_auxiliary_head
        if (
            rgb_resnet_preprocessor_uuid is None
            or depth_resnet_preprocessor_uuid is None
        ):
            resnet_preprocessor_uuid = (
                rgb_resnet_preprocessor_uuid
                if rgb_resnet_preprocessor_uuid is not None
                else depth_resnet_preprocessor_uuid
            )

            self.goal_visual_encoder = ResnetTensorGoalEncoder(
                self.observation_space,
                goal_sensor_uuid,
                resnet_preprocessor_uuid,
                goal_dims,
                resnet_compressor_hidden_out_dims,
                combiner_hidden_out_dims,
            )
        else:
            self.goal_visual_encoder = ResnetDualTensorGoalEncoder(  # type:ignore
                self.observation_space,
                goal_sensor_uuid,
                rgb_resnet_preprocessor_uuid,
                depth_resnet_preprocessor_uuid,
                goal_dims,
                resnet_compressor_hidden_out_dims,
                combiner_hidden_out_dims,
            )
        self.state_encoder = RNNStateEncoder(
            self.goal_visual_encoder.output_dims, self._hidden_size,
        )
        self.actor = LinearActorHead(self._hidden_size, action_space.n)
        self.critic = LinearCriticHead(self._hidden_size)
        if self.include_auxiliary_head:
            self.auxiliary_actor = LinearActorHead(self._hidden_size, action_space.n)
        
        self.embedder = ResNetEmbedder(resnet=resnet_model, pool=pool)

        self.train()
        # TODO!!!: Freeze embedder only


    @property
    def recurrent_hidden_state_size(self) -> int:
        """The recurrent hidden state size of the model."""
        return self._hidden_size

    @property
    def is_blind(self) -> bool:
        """True if the model is blind (e.g. neither 'depth' or 'rgb' is an
        input observation type)."""
        return self.goal_visual_encoder.is_blind

    @property
    def num_recurrent_layers(self) -> int:
        """Number of recurrent hidden layers."""
        return self.state_encoder.num_recurrent_layers

    def _recurrent_memory_specification(self):
        return dict(
            rnn=(
                (
                    ("layer", self.num_recurrent_layers),
                    ("sampler", None),
                    ("hidden", self.recurrent_hidden_state_size),
                ),
                torch.float32,
            )
        )

    def get_object_type_encoding(
        self, observations: Dict[str, torch.FloatTensor]
    ) -> torch.FloatTensor:
        """Get the object type encoding from input batched observations."""
        return self.goal_visual_encoder.get_object_type_encoding(observations)

    def forward(  # type:ignore
        self,
        observations: ObservationType,
        memory: Memory,
        prev_actions: torch.Tensor,
        masks: torch.FloatTensor,
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:
        # if not isinstance(self.preprocessor, ResNetPreprocessor):
        #     raise ValueError(f'self.preprocessor is set to {type(self.preprocessor)}, but should be ResnetPreprocessor')
        
        # # observations[self.rgb_resnet_preprocessor_uuid].shape = torch.Size([1, 15, 224, 224, 3])
        # observations[self.rgb_resnet_preprocessor_uuid] = observations[self.rgb_resnet_preprocessor_uuid].squeeze(0)
        # # Both loops: observations[self.rgb_resnet_preprocessor_uuid].shape = torch.Size([15, 224, 224, 3])
        # observations[self.rgb_resnet_preprocessor_uuid] = self.preprocessor.process(observations)
        # # First loop around: observations[self.rgb_resnet_preprocessor_uuid].shape = torch.Size([1, 15, 224, 224, 3])
        # # Second loop around: observations[self.rgb_resnet_preprocessor_uuid].shape = torch.Size([15, 512, 7, 7])

        # device_visual_encoder = next(self.goal_visual_encoder.parameters()).device
        # if observations[self.rgb_resnet_preprocessor_uuid].is_cuda is not device_visual_encoder:
        #     observations[self.rgb_resnet_preprocessor_uuid] = observations[self.rgb_resnet_preprocessor_uuid].to(device_visual_encoder)

        # Step 1: Make sure RGB observation is resized appropriately for Resnet embedder
        # Step 2: Forward pass through embedder
        observations[self.rgb_resnet_preprocessor_uuid] = self.embedder(observations[self.rgb_resnet_preprocessor_uuid])

        x = self.goal_visual_encoder(observations)
        x, rnn_hidden_states = self.state_encoder(x, memory.tensor("rnn"), masks)
        return (
            ActorCriticOutput(
                distributions=self.actor(x),
                values=self.critic(x),
                extras={"auxiliary_distributions": self.auxiliary_actor(x)}
                if self.include_auxiliary_head
                else {},
            ),
            memory.set_tensor("rnn", rnn_hidden_states),
        )