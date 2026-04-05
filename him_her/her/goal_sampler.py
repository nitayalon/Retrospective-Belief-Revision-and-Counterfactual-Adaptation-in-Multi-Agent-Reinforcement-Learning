"""Goal sampling strategies for Hindsight Experience Replay (HER).

All operations are in NumPy — this runs before JAX conversion in the update step.
"""

import numpy as np
from typing import List

from him_her.models.base_model import Episode


class GoalSampler:
    """Samples alternative goals from episodes for HER relabeling.
    
    Four strategies:
    - future: sample k states from transitions after the current timestep
    - episode: sample k states from anywhere in the episode
    - final: use the final state of the episode (repeated k times)
    - random: sample from the full replay buffer (requires buffer access)
    
    All returned goals are NumPy arrays, not JAX arrays.
    """
    
    STRATEGIES = ["future", "episode", "random", "final"]
    
    def sample_goals(
        self,
        episode: Episode,
        transition_idx: int,
        strategy: str = "future",
        k: int = 4,
        rng: np.random.Generator = None,
    ) -> List[np.ndarray]:
        """Sample k alternative goals from the episode using the specified strategy.
        
        Args:
            episode: Episode object containing transitions
            transition_idx: Index of the transition within the episode (0-indexed)
            strategy: One of ["future", "episode", "final", "random"]
            k: Number of goals to sample
            rng: NumPy random generator (required for "future" and "episode")
        
        Returns:
            List of k NumPy arrays, each representing a goal (achieved state)
        
        Raises:
            ValueError: If strategy is not in STRATEGIES
            NotImplementedError: If strategy is "random" (requires buffer integration)
        
        Note:
            All returned arrays are np.ndarray, not jnp.ndarray.
        """
        if strategy not in self.STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy}'. Must be one of {self.STRATEGIES}"
            )
        
        if strategy == "random":
            raise NotImplementedError(
                "Random strategy requires buffer access and will be implemented "
                "when the replay buffer is integrated."
            )
        
        if strategy == "future":
            return self._sample_future(episode, transition_idx, k, rng)
        elif strategy == "episode":
            return self._sample_episode(episode, k, rng)
        elif strategy == "final":
            return self._sample_final(episode, k)
        
        # Should never reach here due to validation above
        raise ValueError(f"Unhandled strategy: {strategy}")
    
    def _sample_future(
        self,
        episode: Episode,
        transition_idx: int,
        k: int,
        rng: np.random.Generator,
    ) -> List[np.ndarray]:
        """Sample k states from transitions at indices t' > transition_idx.
        
        If fewer than k future states exist, sample with replacement.
        """
        if rng is None:
            raise ValueError("rng is required for 'future' strategy")
        
        episode_length = len(episode.transitions)
        
        # Future transitions are those at indices > transition_idx
        future_indices = list(range(transition_idx + 1, episode_length))
        
        if len(future_indices) == 0:
            # No future states (transition_idx is the last transition)
            # Return the final state repeated k times
            final_state = episode.transitions[-1].next_state
            return [final_state.copy() for _ in range(k)]
        
        # Sample k indices from future transitions (with replacement if needed)
        replace = len(future_indices) < k
        sampled_indices = rng.choice(future_indices, size=k, replace=replace)
        
        # Extract the next_state (achieved state) from each sampled transition
        goals = [episode.transitions[idx].next_state.copy() for idx in sampled_indices]
        
        return goals
    
    def _sample_episode(
        self,
        episode: Episode,
        k: int,
        rng: np.random.Generator,
    ) -> List[np.ndarray]:
        """Sample k states from anywhere in the episode."""
        if rng is None:
            raise ValueError("rng is required for 'episode' strategy")
        
        episode_length = len(episode.transitions)
        
        # Sample k indices from the entire episode (with replacement if needed)
        replace = episode_length < k
        sampled_indices = rng.choice(episode_length, size=k, replace=replace)
        
        # Extract the next_state (achieved state) from each sampled transition
        goals = [episode.transitions[idx].next_state.copy() for idx in sampled_indices]
        
        return goals
    
    def _sample_final(
        self,
        episode: Episode,
        k: int,
    ) -> List[np.ndarray]:
        """Return the final state of the episode repeated k times."""
        final_state = episode.transitions[-1].next_state
        
        # Return k copies of the final state
        return [final_state.copy() for _ in range(k)]
