"""
Pluggable Metrics for Archetype Learning System

This module contains metric implementations for three archetypes:
- Superspreaders: High-impact content creators
- Amplifiers: Influential reshare agents
- Coordinated: Swarm-like behavior patterns

Each metric is a self-contained class that can be plugged into the ArchetypeLearner.
"""

from abc import ABC, abstractmethod
from collections import defaultdict, deque
from typing import TYPE_CHECKING, Dict, List

import numpy as np

if TYPE_CHECKING:
    from .arles import Action, ArchetypeLearner


class ArchetypeMetric(ABC):
    """
    Abstract base class for pluggable archetype metrics.
    Each metric maintains its own state and updates independently.
    """

    @abstractmethod
    def initialize(self, initial_capacity: int) -> None:
        """Initialize metric state with given capacity."""
        pass

    @abstractmethod
    def expand(self, new_capacity: int) -> None:
        """Expand metric arrays to new capacity."""
        pass

    @abstractmethod
    def update(
        self,
        user_idx: int,
        action: "Action",
        timestamp: float,
        learner: "ArchetypeLearner",
    ) -> None:
        """
        Update metric for a user based on an action.

        Args:
            user_idx: User index in the learner's arrays
            action: The action being processed
            timestamp: Action timestamp as float
            learner: Reference to ArchetypeLearner for accessing shared state
        """
        pass

    @abstractmethod
    def get_values(self, n_users: int) -> np.ndarray:
        """
        Get metric values for all users.

        Args:
            n_users: Number of active users

        Returns:
            1D numpy array of metric values
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return metric name for identification."""
        pass


# ============================================================================
# SUPERSPREADER METRICS
# ============================================================================


class AvgResharesPerPost(ArchetypeMetric):
    """Average reshares received per post created."""

    def __init__(self, ema_alpha: float = 0.1):
        self.ema_alpha = ema_alpha
        self.post_count: np.ndarray = np.array([])
        self.total_reshares: np.ndarray = np.array([])
        self.avg_reshares: np.ndarray = np.array([])

    def initialize(self, initial_capacity: int) -> None:
        self.post_count = np.zeros(initial_capacity, dtype=np.float32)
        self.total_reshares = np.zeros(initial_capacity, dtype=np.float32)
        self.avg_reshares = np.zeros(initial_capacity, dtype=np.float32)

    def expand(self, new_capacity: int) -> None:
        self.post_count = np.resize(self.post_count, new_capacity)
        self.total_reshares = np.resize(self.total_reshares, new_capacity)
        self.avg_reshares = np.resize(self.avg_reshares, new_capacity)

    def update(
        self,
        user_idx: int,
        action: "Action",
        timestamp: float,
        learner: "ArchetypeLearner",
    ) -> None:
        if action.activity_type == "post":
            self.post_count[user_idx] = (
                self.ema_alpha * 1.0 + (1 - self.ema_alpha) * self.post_count[user_idx]
            )
            learner.post_to_author[action.action_id] = user_idx

        elif action.activity_type == "repost" and action.original_action_id:
            learner.post_reshares[action.original_action_id].append(
                (user_idx, timestamp)
            )

            if action.original_action_id in learner.post_to_author:
                author_idx = learner.post_to_author[action.original_action_id]
                reshare_count = float(
                    len(learner.post_reshares[action.original_action_id])
                )

                self.total_reshares[author_idx] = (
                    self.ema_alpha * reshare_count
                    + (1 - self.ema_alpha) * self.total_reshares[author_idx]
                )

                if self.post_count[author_idx] > 0:
                    self.avg_reshares[author_idx] = (
                        self.total_reshares[author_idx] / self.post_count[author_idx]
                    )

    def get_values(self, n_users: int) -> np.ndarray:
        return self.avg_reshares[:n_users]

    def get_name(self) -> str:
        return "avg_reshares_per_post"


class PostingFrequency(ArchetypeMetric):
    """Posts per day based on recent activity."""

    def __init__(self, ema_alpha: float = 0.1, max_recent: int = 100):
        self.ema_alpha = ema_alpha
        self.max_recent = max_recent
        self.frequency: np.ndarray = np.array([])
        self.recent_posts: Dict[int, deque] = {}

    def initialize(self, initial_capacity: int) -> None:
        self.frequency = np.zeros(initial_capacity, dtype=np.float32)

    def expand(self, new_capacity: int) -> None:
        self.frequency = np.resize(self.frequency, new_capacity)

    def update(
        self,
        user_idx: int,
        action: "Action",
        timestamp: float,
        learner: "ArchetypeLearner",
    ) -> None:
        if action.activity_type == "post":
            if user_idx not in self.recent_posts:
                self.recent_posts[user_idx] = deque(maxlen=self.max_recent)

            self.recent_posts[user_idx].append(timestamp)

            if len(self.recent_posts[user_idx]) > 1:
                time_span = timestamp - self.recent_posts[user_idx][0]
                if time_span > 0:
                    freq = len(self.recent_posts[user_idx]) / (time_span / 86400)
                    self.frequency[user_idx] = (
                        self.ema_alpha * freq
                        + (1 - self.ema_alpha) * self.frequency[user_idx]
                    )

    def get_values(self, n_users: int) -> np.ndarray:
        return self.frequency[:n_users]

    def get_name(self) -> str:
        return "posting_frequency"


class ViralityScore(ArchetypeMetric):
    """
    Measures how quickly content gets reshared (virality).
    Fast initial reshares indicate viral content creation ability.
    """

    def __init__(self, ema_alpha: float = 0.1, time_threshold: float = 3600):
        """
        Args:
            ema_alpha: EMA smoothing factor
            time_threshold: Time window in seconds to measure early virality (default 1 hour)
        """
        self.ema_alpha = ema_alpha
        self.time_threshold = time_threshold
        self.virality: np.ndarray = np.array([])
        self.post_timestamps: Dict[str, float] = {}

    def initialize(self, initial_capacity: int) -> None:
        self.virality = np.zeros(initial_capacity, dtype=np.float32)

    def expand(self, new_capacity: int) -> None:
        self.virality = np.resize(self.virality, new_capacity)

    def update(
        self,
        user_idx: int,
        action: "Action",
        timestamp: float,
        learner: "ArchetypeLearner",
    ) -> None:
        if action.activity_type == "post":
            self.post_timestamps[action.action_id] = timestamp
            learner.post_to_author[action.action_id] = user_idx

        elif action.activity_type == "repost" and action.original_action_id:
            if action.original_action_id in self.post_timestamps:
                post_time = self.post_timestamps[action.original_action_id]
                time_diff = timestamp - post_time

                # Early reshares get higher score (exponential decay)
                if time_diff <= self.time_threshold:
                    viral_score = np.exp(-time_diff / (self.time_threshold / 3))

                    if action.original_action_id in learner.post_to_author:
                        author_idx = learner.post_to_author[action.original_action_id]
                        self.virality[author_idx] = (
                            self.ema_alpha * viral_score
                            + (1 - self.ema_alpha) * self.virality[author_idx]
                        )

    def get_values(self, n_users: int) -> np.ndarray:
        return self.virality[:n_users]

    def get_name(self) -> str:
        return "virality_score"


def h_index_bisect(rsl: List[int], key=lambda x: x) -> int:
    """Binary search for quick h-index."""
    lo = 0
    hi = len(rsl)

    while lo < hi:
        mid = (lo + hi) // 2
        value = key(rsl[mid])

        if value > mid:
            lo = mid + 1
        elif value < mid:
            hi = mid
        else:
            return mid

    return lo


def vectorized_EMA_reduce(numbers: List[int], alpha: float) -> float:
    """Vectorized Exponential Moving Average."""
    numbers_array = np.asarray(numbers)
    n = len(numbers_array)
    weights = (1 - alpha) * (alpha ** np.arange(n - 1, -1, -1))
    weights[0] = alpha ** (n - 1)
    return float(np.dot(numbers_array, weights))


class TASHIndexMetric(ArchetypeMetric):
    """
    TASH-index metric for superspreaders.
    Tracks social h-index over time slots with EMA smoothing.
    """

    def __init__(self, time_slot_size: float = 15.0, alpha_smoothing: float = 0.4):
        """
        Args:
            time_slot_size: Size of time slot in minutes
            alpha_smoothing: EMA smoothing factor for h-index history
        """
        self.time_slot_size = time_slot_size * 60  # Convert to seconds
        self.alpha_smoothing = alpha_smoothing if alpha_smoothing else 1.0

        self.tash_scores: np.ndarray = np.array([])
        self.current_time_slot = 0
        self.user_track: Dict[int, Dict[str, int]] = {}
        self.user_shi_history: Dict[int, List[int]] = {}
        self.last_flush_time = 0.0

    def initialize(self, initial_capacity: int) -> None:
        self.tash_scores = np.zeros(initial_capacity, dtype=np.float32)

    def expand(self, new_capacity: int) -> None:
        self.tash_scores = np.resize(self.tash_scores, new_capacity)

    def _flush_social_h_index(self, current_time: float) -> None:
        """Compute h-index for all tracked users."""
        for user_idx, track in self.user_track.items():
            track_list = list(track.values())
            track_list.sort(reverse=True)

            shi = h_index_bisect(track_list, key=lambda x: x)

            if user_idx not in self.user_shi_history:
                self.user_shi_history[user_idx] = []
            self.user_shi_history[user_idx].append(shi)

        self.user_track = {}
        self.last_flush_time = current_time

    def update(
        self,
        user_idx: int,
        action: "Action",
        timestamp: float,
        learner: "ArchetypeLearner",
    ) -> None:
        if self.time_slot_size > 0:
            time_slot = int(timestamp // self.time_slot_size)
            if time_slot > self.current_time_slot:
                self._flush_social_h_index(timestamp)
                self.current_time_slot = time_slot

        if action.activity_type == "post":
            if user_idx not in self.user_track:
                self.user_track[user_idx] = {}
            if action.action_id not in self.user_track[user_idx]:
                self.user_track[user_idx][action.action_id] = 0

        elif action.activity_type == "repost" and action.original_action_id:
            if action.original_action_id in learner.post_to_author:
                author_idx = learner.post_to_author[action.original_action_id]

                if author_idx == user_idx:
                    return

                if author_idx not in self.user_track:
                    self.user_track[author_idx] = {}

                if action.original_action_id not in self.user_track[author_idx]:
                    self.user_track[author_idx][action.original_action_id] = 0
                self.user_track[author_idx][action.original_action_id] += 1

            if user_idx not in self.user_track:
                self.user_track[user_idx] = {}

    def get_values(self, n_users: int) -> np.ndarray:
        for user_idx in range(n_users):
            if (
                user_idx in self.user_shi_history
                and len(self.user_shi_history[user_idx]) > 0
            ):
                tash = vectorized_EMA_reduce(
                    self.user_shi_history[user_idx], self.alpha_smoothing
                )
                self.tash_scores[user_idx] = tash

        return self.tash_scores[:n_users]

    def get_name(self) -> str:
        return "tash_index"


# ============================================================================
# AMPLIFIER METRICS
# ============================================================================


class RepostCount(ArchetypeMetric):
    """Total number of reposts made by user."""

    def __init__(self, ema_alpha: float = 0.1):
        self.ema_alpha = ema_alpha
        self.count: np.ndarray = np.array([])

    def initialize(self, initial_capacity: int) -> None:
        self.count = np.zeros(initial_capacity, dtype=np.float32)

    def expand(self, new_capacity: int) -> None:
        self.count = np.resize(self.count, new_capacity)

    def update(
        self,
        user_idx: int,
        action: "Action",
        timestamp: float,
        learner: "ArchetypeLearner",
    ) -> None:
        if action.activity_type == "repost":
            self.count[user_idx] = (
                self.ema_alpha * 1.0 + (1 - self.ema_alpha) * self.count[user_idx]
            )

    def get_values(self, n_users: int) -> np.ndarray:
        return self.count[:n_users]

    def get_name(self) -> str:
        return "repost_count"


class RepostFrequency(ArchetypeMetric):
    """Reposts per day based on recent activity."""

    def __init__(self, ema_alpha: float = 0.1, max_recent: int = 100):
        self.ema_alpha = ema_alpha
        self.max_recent = max_recent
        self.frequency: np.ndarray = np.array([])
        self.recent_reposts: Dict[int, deque] = {}

    def initialize(self, initial_capacity: int) -> None:
        self.frequency = np.zeros(initial_capacity, dtype=np.float32)

    def expand(self, new_capacity: int) -> None:
        self.frequency = np.resize(self.frequency, new_capacity)

    def update(
        self,
        user_idx: int,
        action: "Action",
        timestamp: float,
        learner: "ArchetypeLearner",
    ) -> None:
        if action.activity_type == "repost":
            if user_idx not in self.recent_reposts:
                self.recent_reposts[user_idx] = deque(maxlen=self.max_recent)

            self.recent_reposts[user_idx].append(timestamp)

            if len(self.recent_reposts[user_idx]) > 1:
                time_span = timestamp - self.recent_reposts[user_idx][0]
                if time_span > 0:
                    freq = len(self.recent_reposts[user_idx]) / (time_span / 86400)
                    self.frequency[user_idx] = (
                        self.ema_alpha * freq
                        + (1 - self.ema_alpha) * self.frequency[user_idx]
                    )

    def get_values(self, n_users: int) -> np.ndarray:
        return self.frequency[:n_users]

    def get_name(self) -> str:
        return "repost_frequency"


class WeightedRepostPosition(ArchetypeMetric):
    """Early reposters get higher weight (amplifier influence)."""

    def __init__(self, ema_alpha: float = 0.1, decay_factor: float = 10.0):
        self.ema_alpha = ema_alpha
        self.decay_factor = decay_factor
        self.position: np.ndarray = np.array([])

    def initialize(self, initial_capacity: int) -> None:
        self.position = np.zeros(initial_capacity, dtype=np.float32)

    def expand(self, new_capacity: int) -> None:
        self.position = np.resize(self.position, new_capacity)

    def update(
        self,
        user_idx: int,
        action: "Action",
        timestamp: float,
        learner: "ArchetypeLearner",
    ) -> None:
        if action.activity_type == "repost" and action.original_action_id:
            if action.original_action_id in learner.post_reshares:
                position = len(learner.post_reshares[action.original_action_id])
                weight = np.exp(-position / self.decay_factor)
                self.position[user_idx] = (
                    self.ema_alpha * weight
                    + (1 - self.ema_alpha) * self.position[user_idx]
                )

    def get_values(self, n_users: int) -> np.ndarray:
        return self.position[:n_users]

    def get_name(self) -> str:
        return "weighted_repost_position"


class CascadeAmplification(ArchetypeMetric):
    """
    Measures how much a user's reposts trigger further reshares (cascade effect).
    True amplifiers create secondary waves of sharing.
    """

    def __init__(self, ema_alpha: float = 0.1, cascade_window: float = 7200):
        """
        Args:
            ema_alpha: EMA smoothing factor
            cascade_window: Time window in seconds to track cascades (default 2 hours)
        """
        self.ema_alpha = ema_alpha
        self.cascade_window = cascade_window
        self.amplification: np.ndarray = np.array([])
        self.repost_to_user: Dict[str, int] = {}
        self.repost_timestamps: Dict[str, float] = {}
        self.post_to_reposts: Dict[str, List[str]] = defaultdict(list)

    def initialize(self, initial_capacity: int) -> None:
        self.amplification = np.zeros(initial_capacity, dtype=np.float32)

    def expand(self, new_capacity: int) -> None:
        self.amplification = np.resize(self.amplification, new_capacity)

    def update(
        self,
        user_idx: int,
        action: "Action",
        timestamp: float,
        learner: "ArchetypeLearner",
    ) -> None:
        if action.activity_type == "repost" and action.original_action_id:
            # Track this repost
            self.repost_to_user[action.action_id] = user_idx
            self.repost_timestamps[action.action_id] = timestamp
            self.post_to_reposts[action.original_action_id].append(action.action_id)

            # Check if this is a cascade (repost of a repost)
            if action.original_action_id in self.repost_to_user:
                # This is resharing someone's repost
                original_reposter = self.repost_to_user[action.original_action_id]
                original_repost_time = self.repost_timestamps[action.original_action_id]

                # Check if within cascade window
                time_diff = timestamp - original_repost_time
                if 0 < time_diff <= self.cascade_window:
                    # Credit the original reposter with amplification
                    cascade_score = np.exp(-time_diff / (self.cascade_window / 2))
                    self.amplification[original_reposter] = (
                        self.ema_alpha * cascade_score
                        + (1 - self.ema_alpha) * self.amplification[original_reposter]
                    )

    def get_values(self, n_users: int) -> np.ndarray:
        return self.amplification[:n_users]

    def get_name(self) -> str:
        return "cascade_amplification"


class InfluencerReachScore(ArchetypeMetric):
    """
    Measures if user's reposts reach high-follower users.
    Amplifiers bridge content to influential audiences.
    """

    def __init__(self, ema_alpha: float = 0.1, follower_threshold: int = 100):
        """
        Args:
            ema_alpha: EMA smoothing factor
            follower_threshold: Minimum followers to be considered influential
        """
        self.ema_alpha = ema_alpha
        self.follower_threshold = follower_threshold
        self.reach_score: np.ndarray = np.array([])
        self.user_followers: Dict[int, set] = defaultdict(set)
        self.repost_to_original_author: Dict[str, int] = {}

    def initialize(self, initial_capacity: int) -> None:
        self.reach_score = np.zeros(initial_capacity, dtype=np.float32)

    def expand(self, new_capacity: int) -> None:
        self.reach_score = np.resize(self.reach_score, new_capacity)

    def update(
        self,
        user_idx: int,
        action: "Action",
        timestamp: float,
        learner: "ArchetypeLearner",
    ) -> None:
        if action.activity_type == "repost" and action.original_action_id:
            # Infer follower relationship (reposting suggests following)
            if action.original_action_id in learner.post_to_author:
                original_author = learner.post_to_author[action.original_action_id]
                self.user_followers[original_author].add(user_idx)

                # Track this for cascade analysis
                self.repost_to_original_author[action.action_id] = user_idx

            # Check if current user's followers are resharing
            if user_idx in self.user_followers:
                follower_count = len(self.user_followers[user_idx])

                # Score based on reach (log scale)
                reach = np.log1p(follower_count) / np.log1p(self.follower_threshold)
                reach = min(1.0, reach)

                self.reach_score[user_idx] = (
                    self.ema_alpha * reach
                    + (1 - self.ema_alpha) * self.reach_score[user_idx]
                )

    def get_values(self, n_users: int) -> np.ndarray:
        return self.reach_score[:n_users]

    def get_name(self) -> str:
        return "influencer_reach_score"


class DiverseContentAmplification(ArchetypeMetric):
    """
    Measures diversity of content being amplified.
    Good amplifiers spread diverse content, not just one topic/source.
    """

    def __init__(self, ema_alpha: float = 0.1, window_size: int = 100):
        """
        Args:
            ema_alpha: EMA smoothing factor
            window_size: Number of recent reposts to track
        """
        self.ema_alpha = ema_alpha
        self.window_size = window_size
        self.diversity: np.ndarray = np.array([])
        self.recent_reposts: Dict[int, deque] = {}

    def initialize(self, initial_capacity: int) -> None:
        self.diversity = np.zeros(initial_capacity, dtype=np.float32)

    def expand(self, new_capacity: int) -> None:
        self.diversity = np.resize(self.diversity, new_capacity)

    def update(
        self,
        user_idx: int,
        action: "Action",
        timestamp: float,
        learner: "ArchetypeLearner",
    ) -> None:
        if action.activity_type == "repost" and action.original_action_id:
            if action.original_action_id in learner.post_to_author:
                original_author = learner.post_to_author[action.original_action_id]

                if user_idx not in self.recent_reposts:
                    self.recent_reposts[user_idx] = deque(maxlen=self.window_size)

                self.recent_reposts[user_idx].append(original_author)

                # Calculate diversity (unique authors / total reposts)
                if len(self.recent_reposts[user_idx]) >= 5:
                    unique_authors = len(set(self.recent_reposts[user_idx]))
                    total_reposts = len(self.recent_reposts[user_idx])
                    diversity_score = unique_authors / total_reposts

                    self.diversity[user_idx] = (
                        self.ema_alpha * diversity_score
                        + (1 - self.ema_alpha) * self.diversity[user_idx]
                    )

    def get_values(self, n_users: int) -> np.ndarray:
        return self.diversity[:n_users]

    def get_name(self) -> str:
        return "diverse_content_amplification"


# ============================================================================
# COORDINATED METRICS
# ============================================================================


class CoRepostCount(ArchetypeMetric):
    """Number of users acting on same content within time window."""

    def __init__(
        self,
        ema_alpha: float = 0.1,
        window_seconds: int = 300,
        max_window_size: int = 5000,
    ):
        self.ema_alpha = ema_alpha
        self.window_seconds = window_seconds
        self.max_window_size = max_window_size
        self.count: np.ndarray = np.array([])
        self.recent_actions: deque = deque(maxlen=max_window_size)

    def initialize(self, initial_capacity: int) -> None:
        self.count = np.zeros(initial_capacity, dtype=np.float32)

    def expand(self, new_capacity: int) -> None:
        self.count = np.resize(self.count, new_capacity)

    def update(
        self,
        user_idx: int,
        action: "Action",
        timestamp: float,
        learner: "ArchetypeLearner",
    ) -> None:
        if action.activity_type in ["post", "repost"]:
            content_id = action.original_action_id or action.action_id
            self.recent_actions.append((user_idx, timestamp, content_id))

            cutoff_time = timestamp - self.window_seconds
            coordinated_users = set()

            for other_idx, other_time, other_content in self.recent_actions:
                if (
                    other_time >= cutoff_time
                    and other_idx != user_idx
                    and other_content == content_id
                ):
                    coordinated_users.add(other_idx)

            if len(coordinated_users) > 0:
                self.count[user_idx] = (
                    self.ema_alpha * len(coordinated_users)
                    + (1 - self.ema_alpha) * self.count[user_idx]
                )

    def get_values(self, n_users: int) -> np.ndarray:
        return self.count[:n_users]

    def get_name(self) -> str:
        return "co_repost_count"


class MaxUserSimilarity(ArchetypeMetric):
    """Average similarity with top-k most similar users."""

    def __init__(
        self,
        ema_alpha: float = 0.1,
        top_k: int = 50,
        window_seconds: int = 300,
        max_window_size: int = 5000,
    ):
        self.ema_alpha = ema_alpha
        self.top_k = top_k
        self.window_seconds = window_seconds
        self.max_window_size = max_window_size
        self.similarity: np.ndarray = np.array([])
        self.content_similarity: Dict[int, Dict[int, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self.recent_actions: deque = deque(maxlen=max_window_size)

    def initialize(self, initial_capacity: int) -> None:
        self.similarity = np.zeros(initial_capacity, dtype=np.float32)

    def expand(self, new_capacity: int) -> None:
        self.similarity = np.resize(self.similarity, new_capacity)

    def update(
        self,
        user_idx: int,
        action: "Action",
        timestamp: float,
        learner: "ArchetypeLearner",
    ) -> None:
        if action.activity_type in ["post", "repost"]:
            content_id = action.original_action_id or action.action_id
            self.recent_actions.append((user_idx, timestamp, content_id))

            cutoff_time = timestamp - self.window_seconds

            for other_idx, other_time, other_content in self.recent_actions:
                if (
                    other_time >= cutoff_time
                    and other_idx != user_idx
                    and other_content == content_id
                ):
                    self.content_similarity[user_idx][other_idx] += 1

            if (
                user_idx in self.content_similarity
                and self.content_similarity[user_idx]
            ):
                similarities = list(self.content_similarity[user_idx].values())
                top_k = min(self.top_k, len(similarities))
                top_similarities = sorted(similarities, reverse=True)[:top_k]
                max_sim = float(np.mean(top_similarities))

                self.similarity[user_idx] = (
                    self.ema_alpha * max_sim
                    + (1 - self.ema_alpha) * self.similarity[user_idx]
                )

    def get_values(self, n_users: int) -> np.ndarray:
        return self.similarity[:n_users]

    def get_name(self) -> str:
        return "max_user_similarity"


class ClusteringCoefficient(ArchetypeMetric):
    """Network clustering based on co-reshare patterns."""

    def __init__(
        self,
        ema_alpha: float = 0.1,
        window_seconds: int = 300,
        max_window_size: int = 5000,
    ):
        self.ema_alpha = ema_alpha
        self.window_seconds = window_seconds
        self.max_window_size = max_window_size
        self.coefficient: np.ndarray = np.array([])
        self.content_similarity: Dict[int, Dict[int, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self.recent_actions: deque = deque(maxlen=max_window_size)

    def initialize(self, initial_capacity: int) -> None:
        self.coefficient = np.zeros(initial_capacity, dtype=np.float32)

    def expand(self, new_capacity: int) -> None:
        self.coefficient = np.resize(self.coefficient, new_capacity)

    def update(
        self,
        user_idx: int,
        action: "Action",
        timestamp: float,
        learner: "ArchetypeLearner",
    ) -> None:
        if action.activity_type in ["post", "repost"]:
            content_id = action.original_action_id or action.action_id
            self.recent_actions.append((user_idx, timestamp, content_id))

            cutoff_time = timestamp - self.window_seconds
            coordinated_users = set()

            for other_idx, other_time, other_content in self.recent_actions:
                if (
                    other_time >= cutoff_time
                    and other_idx != user_idx
                    and other_content == content_id
                ):
                    coordinated_users.add(other_idx)
                    self.content_similarity[user_idx][other_idx] += 1

            if len(coordinated_users) >= 2:
                triangle_count = 0
                for u1 in coordinated_users:
                    for u2 in coordinated_users:
                        if u1 < u2 and u2 in self.content_similarity.get(u1, {}):
                            triangle_count += 1

                max_triangles = (
                    len(coordinated_users) * (len(coordinated_users) - 1) / 2
                )
                if max_triangles > 0:
                    clustering = triangle_count / max_triangles
                    self.coefficient[user_idx] = (
                        self.ema_alpha * clustering
                        + (1 - self.ema_alpha) * self.coefficient[user_idx]
                    )

    def get_values(self, n_users: int) -> np.ndarray:
        return self.coefficient[:n_users]

    def get_name(self) -> str:
        return "clustering_coefficient"


class TemporalBurstiness(ArchetypeMetric):
    """
    Measures temporal burstiness of activity.
    Coordinated actors often show synchronized bursts of activity.
    Uses coefficient of variation of inter-event times.
    """

    def __init__(self, ema_alpha: float = 0.1, min_events: int = 10):
        """
        Args:
            ema_alpha: EMA smoothing factor
            min_events: Minimum events needed to compute burstiness
        """
        self.ema_alpha = ema_alpha
        self.min_events = min_events
        self.burstiness: np.ndarray = np.array([])
        self.event_times: Dict[int, deque] = {}

    def initialize(self, initial_capacity: int) -> None:
        self.burstiness = np.zeros(initial_capacity, dtype=np.float32)

    def expand(self, new_capacity: int) -> None:
        self.burstiness = np.resize(self.burstiness, new_capacity)

    def update(
        self,
        user_idx: int,
        action: "Action",
        timestamp: float,
        learner: "ArchetypeLearner",
    ) -> None:
        if action.activity_type in ["post", "repost"]:
            if user_idx not in self.event_times:
                self.event_times[user_idx] = deque(maxlen=100)

            self.event_times[user_idx].append(timestamp)

            # Calculate burstiness if enough events
            if len(self.event_times[user_idx]) >= self.min_events:
                times = np.array(list(self.event_times[user_idx]))
                inter_event_times = np.diff(times)

                if len(inter_event_times) > 0:
                    mean_iet = float(np.mean(inter_event_times))
                    std_iet = float(np.std(inter_event_times))

                    # Coefficient of variation (CV)
                    # High CV = bursty, Low CV = regular
                    if mean_iet > 0:
                        cv = std_iet / mean_iet
                        # Normalize to [0, 1] using tanh
                        burst_score = float(np.tanh(cv / 2))

                        self.burstiness[user_idx] = (
                            self.ema_alpha * burst_score
                            + (1 - self.ema_alpha) * self.burstiness[user_idx]
                        )

    def get_values(self, n_users: int) -> np.ndarray:
        return self.burstiness[:n_users]

    def get_name(self) -> str:
        return "temporal_burstiness"


class SynchronizedActivity(ArchetypeMetric):
    """
    Measures how synchronized a user's activity is with other users.
    Coordinated groups act at similar times on similar content.
    """

    def __init__(
        self, ema_alpha: float = 0.1, sync_window: int = 60, max_window_size: int = 5000
    ):
        """
        Args:
            ema_alpha: EMA smoothing factor
            sync_window: Time window in seconds to consider synchronized
            max_window_size: Maximum actions to track
        """
        self.ema_alpha = ema_alpha
        self.sync_window = sync_window
        self.max_window_size = max_window_size
        self.sync_score: np.ndarray = np.array([])
        self.recent_actions: deque = deque(maxlen=max_window_size)

    def initialize(self, initial_capacity: int) -> None:
        self.sync_score = np.zeros(initial_capacity, dtype=np.float32)

    def expand(self, new_capacity: int) -> None:
        self.sync_score = np.resize(self.sync_score, new_capacity)

    def update(
        self,
        user_idx: int,
        action: "Action",
        timestamp: float,
        learner: "ArchetypeLearner",
    ) -> None:
        if action.activity_type in ["post", "repost"]:
            self.recent_actions.append((user_idx, timestamp, action.activity_type))

            # Count how many other users acted within sync window
            cutoff_time = timestamp - self.sync_window
            sync_count = 0

            for other_idx, other_time, _ in self.recent_actions:
                if other_idx != user_idx and other_time >= cutoff_time:
                    sync_count += 1

            if sync_count > 0:
                # Normalize by log scale
                sync_normalized = float(np.log1p(sync_count) / np.log1p(100))

                self.sync_score[user_idx] = (
                    self.ema_alpha * sync_normalized
                    + (1 - self.ema_alpha) * self.sync_score[user_idx]
                )

    def get_values(self, n_users: int) -> np.ndarray:
        return self.sync_score[:n_users]

    def get_name(self) -> str:
        return "synchronized_activity"


class ContentHomogeneity(ArchetypeMetric):
    """
    Measures how homogeneous (similar) the content shared by a user is.
    Coordinated campaigns often focus on specific narratives/topics.
    Uses vocabulary overlap as a simple proxy.
    """

    def __init__(self, ema_alpha: float = 0.1, min_actions: int = 5):
        """
        Args:
            ema_alpha: EMA smoothing factor
            min_actions: Minimum actions needed to compute homogeneity
        """
        self.ema_alpha = ema_alpha
        self.min_actions = min_actions
        self.homogeneity: np.ndarray = np.array([])
        self.user_content: Dict[int, deque] = {}

    def initialize(self, initial_capacity: int) -> None:
        self.homogeneity = np.zeros(initial_capacity, dtype=np.float32)

    def expand(self, new_capacity: int) -> None:
        self.homogeneity = np.resize(self.homogeneity, new_capacity)

    def update(
        self,
        user_idx: int,
        action: "Action",
        timestamp: float,
        learner: "ArchetypeLearner",
    ) -> None:
        if action.activity_type in ["post", "repost"]:
            # Track which original content user is engaging with
            content_id = action.original_action_id or action.action_id

            if user_idx not in self.user_content:
                self.user_content[user_idx] = deque(maxlen=50)

            self.user_content[user_idx].append(content_id)

            # Calculate homogeneity if enough samples
            if len(self.user_content[user_idx]) >= self.min_actions:
                content_list = list(self.user_content[user_idx])
                unique_content = len(set(content_list))
                total_content = len(content_list)

                # Low diversity = high homogeneity (coordinated campaigns)
                homogeneity_score = 1.0 - (unique_content / total_content)

                self.homogeneity[user_idx] = (
                    self.ema_alpha * homogeneity_score
                    + (1 - self.ema_alpha) * self.homogeneity[user_idx]
                )

    def get_values(self, n_users: int) -> np.ndarray:
        return self.homogeneity[:n_users]

    def get_name(self) -> str:
        return "content_homogeneity"


class ReciprocalInteractionRate(ArchetypeMetric):
    """
    Measures rate of reciprocal interactions between users.
    Coordinated groups show high mutual interaction patterns.
    """

    def __init__(
        self,
        ema_alpha: float = 0.1,
        window_seconds: int = 3600,
        max_window_size: int = 5000,
    ):
        """
        Args:
            ema_alpha: EMA smoothing factor
            window_seconds: Time window to track interactions
            max_window_size: Maximum interactions to track
        """
        self.ema_alpha = ema_alpha
        self.window_seconds = window_seconds
        self.max_window_size = max_window_size
        self.reciprocity: np.ndarray = np.array([])
        self.interactions: deque = deque(maxlen=max_window_size)
        self.user_interactions: Dict[int, Dict[int, int]] = defaultdict(
            lambda: defaultdict(int)
        )

    def initialize(self, initial_capacity: int) -> None:
        self.reciprocity = np.zeros(initial_capacity, dtype=np.float32)

    def expand(self, new_capacity: int) -> None:
        self.reciprocity = np.resize(self.reciprocity, new_capacity)

    def update(
        self,
        user_idx: int,
        action: "Action",
        timestamp: float,
        learner: "ArchetypeLearner",
    ) -> None:
        if action.activity_type == "repost" and action.original_action_id:
            # Track interaction: user_idx reposted from original_author
            if action.original_action_id in learner.post_to_author:
                original_author = learner.post_to_author[action.original_action_id]

                if user_idx != original_author:
                    self.interactions.append((user_idx, original_author, timestamp))
                    self.user_interactions[user_idx][original_author] += 1

                    # Check for reciprocal interaction
                    cutoff_time = timestamp - self.window_seconds
                    reciprocal_count = 0

                    for u_a, u_b, t in self.interactions:
                        if (
                            t >= cutoff_time
                            and u_a == original_author
                            and u_b == user_idx
                        ):
                            reciprocal_count += 1

                    if reciprocal_count > 0:
                        # Measure reciprocity strength
                        total_interactions = self.user_interactions[user_idx][
                            original_author
                        ]
                        reciprocity_score = min(
                            1.0, reciprocal_count / total_interactions
                        )

                        self.reciprocity[user_idx] = (
                            self.ema_alpha * reciprocity_score
                            + (1 - self.ema_alpha) * self.reciprocity[user_idx]
                        )

    def get_values(self, n_users: int) -> np.ndarray:
        return self.reciprocity[:n_users]

    def get_name(self) -> str:
        return "reciprocal_interaction_rate"


class BotLikeBehavior(ArchetypeMetric):
    """
    Detects bot-like patterns: very regular posting intervals,
    high repost rate, low diversity, minimal replies.
    """

    def __init__(self, ema_alpha: float = 0.1, min_events: int = 10):
        """
        Args:
            ema_alpha: EMA smoothing factor
            min_events: Minimum events to compute bot score
        """
        self.ema_alpha = ema_alpha
        self.min_events = min_events
        self.bot_score: np.ndarray = np.array([])
        self.user_events: Dict[int, deque] = {}

    def initialize(self, initial_capacity: int) -> None:
        self.bot_score = np.zeros(initial_capacity, dtype=np.float32)

    def expand(self, new_capacity: int) -> None:
        self.bot_score = np.resize(self.bot_score, new_capacity)

    def update(
        self,
        user_idx: int,
        action: "Action",
        timestamp: float,
        learner: "ArchetypeLearner",
    ) -> None:
        if user_idx not in self.user_events:
            self.user_events[user_idx] = deque(maxlen=100)

        self.user_events[user_idx].append((timestamp, action.activity_type))

        if len(self.user_events[user_idx]) >= self.min_events:
            events = list(self.user_events[user_idx])

            # Feature 1: Regularity of timing (low variance = bot-like)
            timestamps = np.array([e[0] for e in events])
            inter_event_times = np.diff(timestamps)

            if len(inter_event_times) > 0:
                mean_iet = float(np.mean(inter_event_times))
                std_iet = float(np.std(inter_event_times))

                # Low CV = very regular = bot-like
                if mean_iet > 0:
                    cv = std_iet / mean_iet
                    regularity_score = float(
                        1.0 - np.tanh(cv)
                    )  # Invert: low CV = high score
                else:
                    regularity_score = 1.0
            else:
                regularity_score = 0.0

            # Feature 2: High repost ratio
            activity_types = [e[1] for e in events]
            repost_ratio = activity_types.count("repost") / len(activity_types)

            # Feature 3: Low reply ratio (bots don't engage)
            reply_ratio = activity_types.count("reply") / len(activity_types)
            low_engagement = 1.0 - reply_ratio

            # Combine features
            bot_score_combined = (
                0.4 * regularity_score + 0.4 * repost_ratio + 0.2 * low_engagement
            )

            self.bot_score[user_idx] = (
                self.ema_alpha * bot_score_combined
                + (1 - self.ema_alpha) * self.bot_score[user_idx]
            )

    def get_values(self, n_users: int) -> np.ndarray:
        return self.bot_score[:n_users]

    def get_name(self) -> str:
        return "bot_like_behavior"


# ============================================================================
# DEFAULT METRIC SETS
# ============================================================================


def get_default_superspreader_metrics() -> List[ArchetypeMetric]:
    """Get default metrics for superspreader archetype."""
    return [
        AvgResharesPerPost(),
        PostingFrequency(),
        ViralityScore(),
        TASHIndexMetric(),
    ]


def get_default_amplifier_metrics() -> List[ArchetypeMetric]:
    """Get default metrics for amplifier archetype."""
    return [
        RepostCount(),
        RepostFrequency(),
        WeightedRepostPosition(),
        CascadeAmplification(),
        DiverseContentAmplification(),
    ]


def get_default_coordinated_metrics() -> List[ArchetypeMetric]:
    """Get default metrics for coordinated archetype."""
    return [
        CoRepostCount(),
        MaxUserSimilarity(),
        TemporalBurstiness(),
        SynchronizedActivity(),
        BotLikeBehavior(),
    ]


def get_all_available_metrics() -> Dict[str, Dict[str, type]]:
    """Get dictionary of all available metrics by archetype."""
    return {
        "superspreader": {
            "AvgResharesPerPost": AvgResharesPerPost,
            "PostingFrequency": PostingFrequency,
            "ViralityScore": ViralityScore,
            "TASHIndexMetric": TASHIndexMetric,
        },
        "amplifier": {
            "RepostCount": RepostCount,
            "RepostFrequency": RepostFrequency,
            "WeightedRepostPosition": WeightedRepostPosition,
            "CascadeAmplification": CascadeAmplification,
            "InfluencerReachScore": InfluencerReachScore,
            "DiverseContentAmplification": DiverseContentAmplification,
        },
        "coordinated": {
            "CoRepostCount": CoRepostCount,
            "MaxUserSimilarity": MaxUserSimilarity,
            "ClusteringCoefficient": ClusteringCoefficient,
            "TemporalBurstiness": TemporalBurstiness,
            "SynchronizedActivity": SynchronizedActivity,
            "ContentHomogeneity": ContentHomogeneity,
            "ReciprocalInteractionRate": ReciprocalInteractionRate,
            "BotLikeBehavior": BotLikeBehavior,
        },
    }
