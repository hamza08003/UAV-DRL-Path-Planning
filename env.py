import gym
import json
import numpy as np
from gym import spaces
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Any


class UAVEnvironment(gym.Env):
    """
    A Custom Gym environment for UAV navigation including both static and dynamic obstacles.
    It provides observations including UAV position, target position, and obstacle information,
    and computes rewards based on the UAV's progress toward the target and its interactions with obstacles.
    """

    def __init__(self, env_config_filepath: str) -> None:
        """
        Initialize `UAVEnvironment` with the given configuration.
        The configuration is a dictionary loaded from JSON that contains 
        environment parameters, uav start / target positions, and obstacles info.
        
        Args:
            env_config (Dict[str, Any]): Dictionary containing environment parameters.
                Expected keys:
                  - grid_size, start_pos, target_pos, target_radius, max_steps,
                    safe_zone_multiplier, prediction_horizon, base_reward_scale,
                    adaptive_scale_rate, min_reward_scale, max_reward_scale,
                    obstacles.
        """

        super().__init__()

        # load environment config from JSON file
        with open(env_config_filepath, 'r') as file:
            env_config = json.load(file)

        # store environment config dict
        self.config = env_config["environment_config"]

        # environment dimensions and limits
        self.grid_size = tuple(self.config["grid_size"])
        self.max_steps = self.config["max_steps"]
        self.current_step = 0

        # UAV state variables
        self.initial_position = np.array(self.config["start_pos"])
        self.target_position = np.array(self.config["target_pos"])
        self.uav_position = None

        # obstacle management
        self.obstacles = self.config["obstacles"]
        self.static_obstacles = [obs for obs in self.obstacles if "velocity" not in obs]
        self.dynamic_obstacles = [obs for obs in self.obstacles if "velocity" in obs]

        # history tracking
        self.uav_path = []
        self.dynamic_obstacle_history = []
        self.terminal_states = []
        self.terminal_state = None

        # metrics
        self.collision_count = 0
        self.success_count = 0
        self.total_episodes = 0

        # Movement directions (8 possible movements)
        self.directions = {
            'north': [0, 1],
            'south': [0, -1],
            'east': [1, 0],
            'west': [-1, 0],
            'northwest': [-1, 1],
            'northeast': [1, 1],
            'southwest': [-1, -1],
            'southeast': [1, -1]
        }

        # Gym spaces: 8 possible movement directions
        self.action_space = spaces.Discrete(8)
        self.observation_space = self._create_observation_space()

        # reward scaling
        self.reward_scale = self.config["base_reward_scale"]
        self.cumulative_reward = 0.0

        # init first episode
        self.reset()


    def _create_observation_space(self) -> spaces.Dict:
        """
        Creates the observation space dictionary with all observable state variables.

        Returns:
            spaces.Dict: A dictionary space with keys:
            'uav_pos', 'target_pos', 'static_obstacles', and 'dynamic_obstacles'.
        """
        
        prediction_horizon = self.config["prediction_horizon"]
        grid_max = max(self.grid_size)

        return spaces.Dict({
            'uav_pos': spaces.Box(
                low=0,
                high=grid_max,
                shape=(2,),
                dtype=np.float32
            ),
            'target_pos': spaces.Box(
                low=0,
                high=grid_max,
                shape=(2,),
                dtype=np.float32
            ),
            'static_obstacles': spaces.Sequence(
                spaces.Dict({
                    'pos': spaces.Box(
                        low=0,
                        high=grid_max,
                        shape=(2,),
                        dtype=np.float32
                    ),
                    'radius': spaces.Box(
                        low=0,
                        high=grid_max,
                        shape=(1,),
                        dtype=np.float32
                    )
                })
            ),
            'dynamic_obstacles': spaces.Sequence(
                spaces.Dict({
                    'pos': spaces.Box(
                        low=0,
                        high=grid_max,
                        shape=(2,),
                        dtype=np.float32
                    ),
                    'velocity': spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(2,),
                        dtype=np.float32
                    ),
                    'predicted_positions': spaces.Box(
                        low=0,
                        high=grid_max,
                        shape=(prediction_horizon, 2),
                        dtype=np.float32
                    )
                })
            )
        })


    def reset(self) -> Dict[str, np.ndarray]:
        """
        Resets the environment to its initial state.

        Returns:
            Dict[str, np.ndarray]: The initial observation dictionary.
        """

        self.current_step = 0
        self.cumulative_reward = 0.0
        self.terminal_state = None

        # reset UAV position and path
        self.uav_position = self.initial_position.copy()
        self.uav_path = [self.uav_position.copy()]

        # reset dynamic obstacles to initial positions
        for obstacle in self.dynamic_obstacles:
            if isinstance(obstacle['range'][0], int):
                obstacle['pos'] = [obstacle['range'][0], obstacle['pos'][1]]
            else:
                obstacle['pos'] = [obstacle['range'][0][0], obstacle['range'][1][0]]

        # clear history for new episode
        self.dynamic_obstacle_history = []
        # create initial observation
        observation = self._get_observation()
        
        self.total_episodes += 1
        return observation


    def _get_observation(self) -> Dict[str, np.ndarray]:
        """
        Creates and retrieves the current observation dictionary from the environment's state.

        Returns:
            Dict[str, np.ndarray]: Dictionary containing the UAV position, target position,
            static obstacles, and dynamic obstacles (with predicted positions).
        """

        observation = {
            'uav_pos': self.uav_position.astype(np.float32),
            'target_pos': self.target_position.astype(np.float32),
            'static_obstacles': [{
                'pos': np.array(obs['pos'], dtype=np.float32),
                'radius': np.array([obs['rad']], dtype=np.float32)
            } for obs in self.static_obstacles],
            'dynamic_obstacles': []
        }

        # add dynamic obstacles with predictions
        for obs in self.dynamic_obstacles:
            predicted_positions = self._predict_obstacle_trajectory(obs)
            observation['dynamic_obstacles'].append({
                'pos': np.array(obs['pos'], dtype=np.float32),
                'velocity': np.array(obs['velocity'], dtype=np.float32),
                'predicted_positions': predicted_positions
            })

        return observation


    def _predict_obstacle_trajectory(self, obstacle: Dict[str, Any]) -> np.ndarray:
        """
        Calculates future positions of a dynamic obstacle.

        Args:
            obstacle (Dict[str, Any]): Dictionary containing obstacle properties.

        Returns:
            np.ndarray: Array of predicted positions with shape
            (prediction_horizon, 2).
        """

        prediction_horizon = self.config["prediction_horizon"]
        predictions = np.zeros((prediction_horizon, 2), dtype=np.float32)
        current_pos = np.array(obstacle['pos'], dtype=np.float32)
        velocity = np.array(obstacle['velocity'], dtype=np.float32)

        for step in range(self.config["prediction_horizon"]):
            next_pos = current_pos + velocity

            # handle boundary conditions
            if isinstance(obstacle['range'][0], int):
                min_range, max_range = obstacle['range']
                if next_pos[0] > max_range or next_pos[0] < min_range:
                    velocity[0] *= -1
                next_pos[0] = np.clip(next_pos[0], min_range, max_range)
            else:
                (min_x, max_x), (min_y, max_y) = obstacle['range']
                if next_pos[0] > max_x or next_pos[0] < min_x:
                    velocity[0] *= -1
                if next_pos[1] > max_y or next_pos[1] < min_y:
                    velocity[1] *= -1
                next_pos = np.clip(next_pos, [min_x, min_y], [max_x, max_y])

            predictions[step] = next_pos
            current_pos = next_pos

        return predictions


    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        """
        Executes one step in the environment based on the given action.

        Args:
            action (int): Integer representing the action to take (0-7 corresponding to movement directions).

        Returns:
            Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
                - observation: Updated state observation.
                - reward: Reward for the current step.
                - done: Boolean flag indicating whether the episode has ended.
                - info: Dictionary with additional information (e.g., terminal state, cumulative reward, etc.).
        """

        self.current_step += 1

        # get movement direction
        movement = list(self.directions.values())[action]

        # update UAV position
        old_position = self.uav_position.copy()
        self.uav_position = np.clip(
            self.uav_position + movement,
            0,
            np.array(self.grid_size) - 1
        )

        # update path history
        self.uav_path.append(self.uav_position.copy())
        # move dynamic obstacles
        self._move_dynamic_obstacles()
        
        # calculate reward and check if episode is done
        reward = self._calculate_reward(old_position)
        self.cumulative_reward += reward
        done = self._is_done()

        # update episode stats
        if done:
            if self.terminal_state == 'finish':
                self.success_count += 1
            elif self.terminal_state == 'collision':
                self.collision_count += 1
            self.terminal_states.append(self.terminal_state)

        # create info dictionary
        info = {
            'terminal_state': self.terminal_state if done else None,
            'cumulative_reward': self.cumulative_reward,
            'success_rate': self.success_count / self.total_episodes,
            'collision_rate': self.collision_count / self.total_episodes,
            'step_count': self.current_step
        }

        return self._get_observation(), reward, done, info


    def _move_dynamic_obstacles(self) -> None:
        """
        Updates positions of all dynamic obstacles based on their velocities and range constraints.
        """

        for obstacle in self.dynamic_obstacles:
            pos = np.array(obstacle['pos'])
            velocity = np.array(obstacle['velocity'])

            # update position based on movement constraints
            if isinstance(obstacle['range'][0], int):
                min_range, max_range = obstacle['range']
                if pos[0] + velocity[0] > max_range or pos[0] + velocity[0] < min_range:
                    velocity[0] *= -1
                    obstacle['velocity'][0] *= -1
            else:
                (min_x, max_x), (min_y, max_y) = obstacle['range']
                if pos[0] + velocity[0] > max_x or pos[0] + velocity[0] < min_x:
                    velocity[0] *= -1
                    obstacle['velocity'][0] *= -1
                if pos[1] + velocity[1] > max_y or pos[1] + velocity[1] < min_y:
                    velocity[1] *= -1
                    obstacle['velocity'][1] *= -1

            # update position
            obstacle['pos'] = (pos + velocity).tolist()

            # record encounter data
            distance: float = np.linalg.norm(self.uav_position - pos)
            self.dynamic_obstacle_history.append({
                'distance': distance,
                'radius': obstacle['rad'],
                'relative_velocity': velocity,
                'position': pos.copy()
            })


    def _calculate_reward(self, old_position: np.ndarray) -> float:
        """
        Calculates the reward for the current step based on movement toward the target,
        safe zone violations, collisions, and movement efficiency.

        Args:
            old_position (np.ndarray): UAV position before taking the step.

        Returns:
            float: Calculated reward for the current step.
        """

        reward = 0.0

        # 1. Distance to goal reward: reward for moving closer to the goal
        old_distance = np.linalg.norm(old_position - self.target_position)
        new_distance = np.linalg.norm(self.uav_position - self.target_position)
        reward += 0.1 * (old_distance - new_distance)  # suggested: 0.2 to 0.5

        # 2. Goal achievement reward
        if new_distance <= self.config["target_radius"]:
            reward += 100.0
            return self._adaptive_reward_scaling(reward)

        # 3. Safe zone violation penalty
        safe_zone_violation = self._calculate_safe_zone_violation(self.uav_position)
        reward -= 20.0 * safe_zone_violation  # suggested: -10.0 to -20.0

        # 4. Collision penalties
        for obstacle in self.obstacles:
            distance = np.linalg.norm(self.uav_position - np.array(obstacle['pos']))
            if distance < obstacle['rad']:
                reward -= 50.0

        # 5. Movement efficiency reward: reward for aligning movement with direction to goal
        if len(self.uav_path) >= 2:
            movement_vector = self.uav_position - old_position
            goal_vector = self.target_position - self.uav_position
            # avoid division by zero
            if np.linalg.norm(movement_vector) > 0 and np.linalg.norm(goal_vector) > 0:
                alignment = np.dot(movement_vector, goal_vector) / (np.linalg.norm(movement_vector) * np.linalg.norm(goal_vector))
            else:
                alignment = 0.0
            reward += 5.0 * alignment  # suggested: 8.0 to 10.0

        return self._adaptive_reward_scaling(reward)


    def _calculate_safe_zone_violation(self, position: np.ndarray) -> float:
        """
        Calculates the degree of safe zone violation for the given UAV position.

        Args:
            position (np.ndarray): The UAV's current position.

        Returns:
            float: Violation score in the range [0, 1], where higher values indicate deeper
            intrusion into a safe zone.
        """

        violation = 0.0
        for obstacle in self.obstacles:
            distance = np.linalg.norm(position - np.array(obstacle['pos']))
            safe_radius = obstacle["rad"] * self.config["safe_zone_multiplier"]
            if safe_radius > 0 and distance < safe_radius:
                violation += (safe_radius - distance) / safe_radius
        return min(violation, 1.0)


    def _adaptive_reward_scaling(self, base_reward: float) -> float:
        """
        Scales the reward based on the recent performance of the UAV (i.e., success rate).

        Args:
            base_reward (float): The original reward value.

        Returns:
            float: The scaled reward value.
        """

        if len(self.terminal_states) > 100:
            recent_success = np.mean([
                1 if state == 'finish' else 0
                for state in self.terminal_states[-100:]
            ])
            target_success = 0.7

            error = target_success - recent_success
            self.reward_scale += self.config["adaptive_scale_rate"] * error
            self.reward_scale = np.clip(
                self.reward_scale,
                self.config["min_reward_scale"],
                self.config["max_reward_scale"]
            )

        return base_reward * self.reward_scale


    def _is_done(self) -> bool:
        """
        Checks whether the episode should end.

        Returns:
            bool: True if the episode is done, otherwise False.
        """

        # 1. Check goal achievement
        if np.linalg.norm(self.uav_position - self.target_position) <= self.config["target_radius"]:
            self.terminal_state = 'finish'
            return True

        # 2. Check collisions
        for obstacle in self.obstacles:
            if np.linalg.norm(self.uav_position - np.array(obstacle['pos'])) < obstacle['rad']:
                self.terminal_state = 'collision'
                return True

        # 3 Check step limit
        if self.current_step >= self.max_steps:
            self.terminal_state = 'out_of_steps'
            return True

        return False


    def render(self, mode: str = 'human') -> None:
        """
        Renders the current state of the environment including the UAV's path,
        obstacles, target, and predicted trajectories for dynamic obstacles.

        Args:
            mode (str): The rendering mode (e.g., 'human' for matplotlib visualization).
        """

        plt.figure(figsize=(10, 10))
        plt.xlim(0, self.grid_size[0])
        plt.ylim(0, self.grid_size[1])
        plt.title("UAV Environment")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")

        # plot UAV path
        path = np.array(self.uav_path)
        plt.plot(path[:, 0], path[:, 1], 'b.-', label='UAV Path')

        # plot current UAV position
        plt.plot(self.uav_position[0], self.uav_position[1], 'bo', markersize=8, label='UAV')

        # plot target position
        plt.plot(self.target_position[0], self.target_position[1], 'g*', markersize=12, label='Target')

        # plot static obstacles
        for obs in self.static_obstacles:
            circle = plt.Circle(tuple(obs['pos']), obs['rad'], color='r', fill=False, label='Static Obstacle')
            plt.gca().add_artist(circle)

        # plot dynamic obstacles and their predicted trajectories
        for idx, obs in enumerate(self.dynamic_obstacles):
            # current dynamic obstacle
            circle = plt.Circle(tuple(obs['pos']), obs['rad'], color='m', fill=False, label='Dynamic Obstacle' if idx == 0 else "")
            plt.gca().add_artist(circle)

            # predicted trajectory from current observation
            # using _predict_obstacle_trajectory to get the predicted positions
            predicted_positions = self._predict_obstacle_trajectory(obs)
            plt.plot(predicted_positions[:, 0],
                     predicted_positions[:, 1],
                     'k--',
                     label='Predicted Trajectory' if idx == 0 else "")

        plt.legend()
        plt.show()
