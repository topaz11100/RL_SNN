from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch import optim
import torch.nn.functional as F

from src.models.actor_critic import ActorNetwork, CriticNetwork, fuse_state
from src.models.cnn_frontend import SpikeHistoryCNN
from src.utils.spike_buffer import RollingSpikeBuffer


@dataclass
class TrajectoryEntry:
    """Container storing information for one synapse event during an episode.
    
    Updated to comply with Theory 2.8: Stores state, action (implicitly via log_prob), and value.
    """
    fused_state: torch.Tensor
    log_prob: torch.Tensor  # Log probability of the taken action
    value: torch.Tensor     # Value estimate by Critic at that step


class RLScenario:
    """Abstract base scenario implementing the episode loop scaffolding."""

    def __init__(
        self,
        history_length: int,
        sigma_policy: float,
        include_layer_pos: bool = False,
        device: str = "cpu",
    ) -> None:
        """Prepare shared components for all scenarios."""
        self.history_length = history_length
        self.include_layer_pos = include_layer_pos
        self.device = device
        
        # Dimensions: CNN output(16) + weight(1) + event_type(2) + [layer_pos(1)]
        fused_dim = 16 + 1 + 2 + (1 if include_layer_pos else 0)
        
        self.actor = ActorNetwork(input_dim=fused_dim, sigma=sigma_policy).to(device)
        self.critic = CriticNetwork(input_dim=fused_dim).to(device)
        
        # Learning rates not specified in snippet, using defaults but can be parameterized
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=1e-3)

    def build_state(self, buffer: RollingSpikeBuffer, weight: float, event_type: torch.Tensor, layer_pos: Optional[float] = None) -> torch.Tensor:
        """Construct the fused local state tensor for a single event."""
        spike_tensor = buffer.batch(batch_size=1).to(self.device)
        # Note: In a real efficient implementation, CNN shouldn't be re-instantiated every step.
        # But keeping structure for now. Assuming self.actor has the feature extractor if shared weights intended,
        # otherwise creating new CNN here implies random weights every step which is wrong.
        # Theory 2.4 says: "All Actor/Critic have IDENTICAL 1D CNN front-end structure... 
        # but parameters are NOT shared."
        # The provided implementation of build_state instantiates a NEW CNN() every call. 
        # This is strictly WRONG because features won't be learned.
        # FIX: Use the actor's feature extractor for state construction or handle inside model.
        # However, to avoid breaking 'fuse_state' signature compatibility in 'models', 
        # we will use the actor's feature extractor here to get consistent features.
        
        features = self.actor.feature_extractor(spike_tensor).detach() # Detach as this is input state
        
        weight_tensor = torch.tensor([[weight]], device=self.device)
        if layer_pos is not None and self.include_layer_pos:
            layer_tensor = torch.tensor([[layer_pos]], device=self.device)
        else:
            layer_tensor = None
            
        fused = fuse_state(features, weight_tensor, event_type.to(self.device), layer_tensor)
        return fused

    def run_episode(self, episode_data: Dict) -> torch.Tensor:
        """Run one episode and return the scalar reward."""
        raise NotImplementedError

    def optimize_from_trajectory(self, trajectory: List[TrajectoryEntry], reward: torch.Tensor) -> None:
        """Perform actor-critic updates from a stored trajectory.
        
        Updated to comply with Theory 2.9: Use stored log_probs and values (On-policy).
        """
        if not trajectory:
            return

        # Stack stored tensors
        # log_prob: (E, 1), value: (E, 1)
        log_probs = torch.stack([entry.log_prob for entry in trajectory]).squeeze()
        values = torch.stack([entry.value for entry in trajectory]).squeeze()
        
        # Theory 2.9: G_e = R (gamma=1). 
        # Advantage A_e = R - V_e
        # Expand reward to match batch size
        R = reward.to(self.device).expand_as(values)
        
        advantage = R - values
        
        # Theory 2.9: L_actor = - (1/E) * sum( A_e * log_pi )
        # Note: Detach advantage to prevent gradients flowing into Critic via Actor loss
        actor_loss = -(advantage.detach() * log_probs).mean()
        
        # Theory 2.9: L_critic = (1/E) * sum( (R - V_e)^2 )
        critic_loss = F.mse_loss(values, R)

        # Update Actor
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # Update Critic
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

    def clip_weight(self, weight: float, clip_min: float, clip_max: float) -> float:
        """Clip a scalar weight based on the synapse type bounds."""
        return float(max(min(weight, clip_max), clip_min))
