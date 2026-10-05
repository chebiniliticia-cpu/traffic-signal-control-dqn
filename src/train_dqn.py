import random
import time
from collections import deque

import numpy as np
import tensorflow as tf
import traci


# SUMO settings
SUMO_CONFIG = r"C:\Users\ASUS\Desktop\traffic_dql\traffic_dql.sumocfg"
TLS_ID = "319612951"

LANES_GROUP_1 = ["-371466059_0", "29057968_0"]
LANES_GROUP_2 = ["29057921_0", "-371466060_0"]

TLS_GREEN = {
    0: "GGggrrrrGGggrrrr",
    1: "rrrrGGggrrrrGGgg"
}

TLS_YELLOW = {
    0: "yyyyrrrryyyyrrrr",
    1: "rrrryyyyrrrryyyy"
}


# Timing
TIME_SLOT = 10
YELLOW_TIME = 4
MIN_GREEN_TIME = 10

STEPS_PER_EPISODE = 180
EPISODES = 300


# State
MAX_QUEUE = 30.0

STATE_SIZE = 3
ACTION_SIZE = 2


def normalize_state(q1, q2, light):
    return np.array(
        [q1 / MAX_QUEUE, q2 / MAX_QUEUE, float(light)],
        dtype=np.float32
    )


# DQN parameters
HIDDEN_SIZE = 64
BATCH_SIZE = 125

GAMMA = 0.99
LEARNING_RATE = 0.001

REPLAY_MEMORY_SIZE = 5000

EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.98

UPDATE_TARGET_EVERY = 10

MODEL_PATH = (
    r"C:\Users\ASUS\Desktop\traffic_dql"
    r"\dqn_single_intersection_v2.keras"
)


# Reproducibility
SEED = 43

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


class TrafficEnvironment:

    def __init__(self):
        self.tls = TLS_ID
        self.lanes_g1 = LANES_GROUP_1
        self.lanes_g2 = LANES_GROUP_2

        self.light = 0
        self.time_since_change = 0

    def reset(self):
        self.light = 0
        self.time_since_change = 0

        traci.trafficlight.setRedYellowGreenState(
            self.tls,
            TLS_GREEN[self.light]
        )

        return self.get_state()

    def get_queues(self):
        q1 = sum(
            traci.lane.getLastStepHaltingNumber(lane)
            for lane in self.lanes_g1
        )

        q2 = sum(
            traci.lane.getLastStepHaltingNumber(lane)
            for lane in self.lanes_g2
        )

        return q1, q2

    def get_state(self):
        q1, q2 = self.get_queues()
        return normalize_state(q1, q2, self.light)

    def step(self, action):

        if action == 1 and self.time_since_change >= MIN_GREEN_TIME:

            traci.trafficlight.setRedYellowGreenState(
                self.tls,
                TLS_YELLOW[self.light]
            )

            for _ in range(YELLOW_TIME):
                traci.simulationStep()

            self.light = 1 - self.light
            self.time_since_change = 0

        else:
            self.time_since_change += TIME_SLOT

        traci.trafficlight.setRedYellowGreenState(
            self.tls,
            TLS_GREEN[self.light]
        )

        for _ in range(TIME_SLOT):
            traci.simulationStep()

        q1, q2 = self.get_queues()

        reward = -(q1 ** 2 + q2 ** 2)

        next_state = normalize_state(q1, q2, self.light)

        done = False

        return next_state, reward, done


class DQNAgent:

    def __init__(self, state_size, action_size):

        self.state_size = state_size
        self.action_size = action_size

        self.memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.gamma = GAMMA

        self.epsilon = EPSILON_START
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY

        self.model = self.build_model()
        self.target_model = self.build_model()

        self.update_target_network()

    def build_model(self):

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.state_size,)),
            tf.keras.layers.Dense(HIDDEN_SIZE, activation="relu"),
            tf.keras.layers.Dense(HIDDEN_SIZE, activation="relu"),
            tf.keras.layers.Dense(self.action_size, activation="linear")
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=LEARNING_RATE
            ),
            loss="mse"
        )

        return model

    def update_target_network(self):
        self.target_model.set_weights(
            self.model.get_weights()
        )

    def act(self, state):

        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        q_values = self.model.predict(
            state.reshape(1, -1),
            verbose=0
        )

        return int(np.argmax(q_values[0]))

    def remember(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):
        self.memory.append(
            (state, action, reward, next_state, done)
        )

    def replay(self):

        if len(self.memory) < BATCH_SIZE:
            return None

        batch = random.sample(
            self.memory,
            BATCH_SIZE
        )

        states = np.array(
            [x[0] for x in batch],
            dtype=np.float32
        )

        actions = np.array(
            [x[1] for x in batch],
            dtype=np.int32
        )

        rewards = np.array(
            [x[2] for x in batch],
            dtype=np.float32
        )

        next_states = np.array(
            [x[3] for x in batch],
            dtype=np.float32
        )

        dones = np.array(
            [x[4] for x in batch],
            dtype=np.float32
        )

        current_q = self.model.predict(
            states,
            verbose=0
        )

        next_q = self.target_model.predict(
            next_states,
            verbose=0
        )

        max_next_q = np.max(
            next_q,
            axis=1
        )

        targets = current_q.copy()

        for i in range(BATCH_SIZE):

            if dones[i]:
                targets[i, actions[i]] = rewards[i]

            else:
                targets[i, actions[i]] = (
                    rewards[i]
                    + self.gamma * max_next_q[i]
                )

        history = self.model.fit(
            states,
            targets,
            epochs=1,
            verbose=0
        )

        return history.history["loss"][0]

    def decay_epsilon(self):
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )


def close_sumo(label):

    try:
        traci.switch(label)
        traci.close()

    except Exception as e:
        print(f"SUMO close error: {e}")

    time.sleep(0.5)


def train():

    env = TrafficEnvironment()
    agent = DQNAgent(
        STATE_SIZE,
        ACTION_SIZE
    )

    episode_rewards = []

    print("=" * 50)
    print("DQN - Traffic Light Control")
    print("=" * 50)

    for episode in range(1, EPISODES + 1):

        label = f"episode_{episode}"

        print(
            f"\nEpisode {episode}/{EPISODES}"
        )

        traci.start(
            ["sumo", "-c", SUMO_CONFIG],
            label=label
        )

        traci.switch(label)

        try:

            state = env.reset()
            total_reward = 0.0

            for step in range(
                1,
                STEPS_PER_EPISODE + 1
            ):

                action = agent.act(state)

                next_state, reward, done = env.step(
                    action
                )

                agent.remember(
                    state,
                    action,
                    reward,
                    next_state,
                    done
                )

                loss = agent.replay()

                state = next_state
                total_reward += reward

                if step % 10 == 0:

                    loss_text = (
                        f"{loss:.6f}"
                        if loss is not None
                        else "N/A"
                    )

                    print(
                        f"Step {step:3d} | "
                        f"Action {action} | "
                        f"Light {env.light} | "
                        f"Reward {reward:8.2f} | "
                        f"Loss {loss_text}"
                    )

            episode_rewards.append(
                total_reward
            )

            agent.decay_epsilon()

            if episode % UPDATE_TARGET_EVERY == 0:

                agent.update_target_network()

                print(
                    "Target network updated."
                )

            print(
                f"Total reward: {total_reward:.2f}"
            )

            print(
                f"Epsilon: {agent.epsilon:.4f}"
            )

            print(
                f"Memory: {len(agent.memory)}"
            )

        finally:
            close_sumo(label)

    agent.model.save(MODEL_PATH)

    print("\nTraining finished.")
    print(f"Model saved to: {MODEL_PATH}")

    return agent, episode_rewards


if __name__ == "__main__":

    agent, rewards = train()
