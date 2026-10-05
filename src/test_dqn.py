import csv
import time

import numpy as np
import tensorflow as tf
import traci


SUMO_CONFIG = (
    r"C:\Users\ASUS\Desktop\traffic_dql"
    r"\traffic_dql.sumocfg"
)

TLS_ID = "319612951"

LANES_GROUP_1 = [
    "-371466059_0",
    "29057968_0"
]

LANES_GROUP_2 = [
    "29057921_0",
    "-371466060_0"
]


TLS_GREEN = {
    0: "GGggrrrrGGggrrrr",
    1: "rrrrGGggrrrrGGgg"
}

TLS_YELLOW = {
    0: "yyyyrrrryyyyrrrr",
    1: "rrrryyyyrrrryyyy"
}


TIME_SLOT = 10
YELLOW_TIME = 4
MIN_GREEN_TIME = 10

TOTAL_SIM_TIME = 1800
MAX_QUEUE = 30.0


MODEL_PATH = (
    r"C:\Users\ASUS\Desktop\traffic_dql"
    r"\dqn_single_intersection_v2.keras"
)

OUTPUT_CSV = (
    r"C:\Users\ASUS\Desktop\traffic_dql"
    r"\test_dqn_timeseries_v2.csv"
)

SUMO_BINARY = "sumo"


def normalize_state(q1, q2, light):

    return np.array(
        [
            q1 / MAX_QUEUE,
            q2 / MAX_QUEUE,
            float(light)
        ],
        dtype=np.float32
    )


class TestEnvironment:

    def __init__(self):

        self.tls = TLS_ID
        self.lanes_g1 = LANES_GROUP_1
        self.lanes_g2 = LANES_GROUP_2

        self.light = 0
        self.time_since_change = 0

        self.depart_time = {}
        self.waiting_time_acc = {}

        self.completed_travel_times = []
        self.completed_waiting_times = []

        self.total_departed = 0
        self.total_arrived = 0

        self.previous_speed = {}
        self.total_stops = 0

        self.time_history = []
        self.queue_g1_history = []
        self.queue_g2_history = []
        self.queue_total_history = []
        self.waiting_count_history = []
        self.vehicles_present_history = []
        self.phase_history = []
        self.total_waiting_time_history = []
        self.arrived_history = []
        self.departed_history = []
        self.throughput_history = []
        self.avg_waiting_time_history = []
        self.stops_history = []

    def reset(self):

        self.light = 0
        self.time_since_change = 0

        traci.trafficlight.setRedYellowGreenState(
            self.tls,
            TLS_GREEN[self.light]
        )

        q1, q2 = self.read_queues()

        return normalize_state(
            q1,
            q2,
            self.light
        )

    def read_queues(self):

        q1 = sum(
            traci.lane.getLastStepHaltingNumber(lane)
            for lane in self.lanes_g1
        )

        q2 = sum(
            traci.lane.getLastStepHaltingNumber(lane)
            for lane in self.lanes_g2
        )

        return q1, q2

    def log_one_second(self):

        traci.simulationStep()

        current_time = traci.simulation.getTime()

        for vid in traci.simulation.getDepartedIDList():

            self.depart_time[vid] = current_time
            self.waiting_time_acc[vid] = 0.0
            self.total_departed += 1

        present_ids = traci.vehicle.getIDList()

        waiting_now = 0

        for vid in present_ids:

            speed = traci.vehicle.getSpeed(vid)

            if speed < 0.1:

                waiting_now += 1

                self.waiting_time_acc[vid] = (
                    self.waiting_time_acc.get(vid, 0.0)
                    + 1.0
                )

            previous_speed = self.previous_speed.get(
                vid,
                100.0
            )

            if previous_speed >= 0.1 and speed < 0.1:
                self.total_stops += 1

            self.previous_speed[vid] = speed

        for vid in traci.simulation.getArrivedIDList():

            depart = self.depart_time.get(
                vid,
                current_time
            )

            travel_time = current_time - depart

            waiting_time = self.waiting_time_acc.get(
                vid,
                0.0
            )

            self.completed_travel_times.append(
                travel_time
            )

            self.completed_waiting_times.append(
                waiting_time
            )

            self.total_arrived += 1

            self.depart_time.pop(vid, None)
            self.waiting_time_acc.pop(vid, None)
            self.previous_speed.pop(vid, None)

        q1, q2 = self.read_queues()

        queue_total = q1 + q2

        previous_waiting_time = (
            self.total_waiting_time_history[-1]
            if self.total_waiting_time_history
            else 0.0
        )

        current_waiting_time = (
            previous_waiting_time
            + waiting_now
        )

        if self.total_departed > 0:

            average_waiting_time = (
                current_waiting_time
                / self.total_departed
            )

        else:

            average_waiting_time = 0.0

        if current_time > 0:

            throughput = (
                self.total_arrived
                / current_time
                * 3600.0
            )

        else:

            throughput = 0.0

        self.time_history.append(current_time)
        self.queue_g1_history.append(q1)
        self.queue_g2_history.append(q2)
        self.queue_total_history.append(queue_total)

        self.waiting_count_history.append(
            waiting_now
        )

        self.vehicles_present_history.append(
            len(present_ids)
        )

        self.phase_history.append(self.light)

        self.total_waiting_time_history.append(
            current_waiting_time
        )

        self.arrived_history.append(
            self.total_arrived
        )

        self.departed_history.append(
            self.total_departed
        )

        self.throughput_history.append(
            throughput
        )

        self.avg_waiting_time_history.append(
            average_waiting_time
        )

        self.stops_history.append(
            self.total_stops
        )

        return q1, q2

    def step(self, action):

        can_change = (
            self.time_since_change
            >= MIN_GREEN_TIME
        )

        if action == 1 and can_change:

            traci.trafficlight.setRedYellowGreenState(
                self.tls,
                TLS_YELLOW[self.light]
            )

            for _ in range(YELLOW_TIME):
                self.log_one_second()

            self.light = 1 - self.light
            self.time_since_change = 0

        else:

            self.time_since_change += TIME_SLOT

        traci.trafficlight.setRedYellowGreenState(
            self.tls,
            TLS_GREEN[self.light]
        )

        q1, q2 = 0, 0

        for _ in range(TIME_SLOT):
            q1, q2 = self.log_one_second()

        return normalize_state(
            q1,
            q2,
            self.light
        )


def close_sumo(label):

    print(
        ">>> Closing SUMO...",
        flush=True
    )

    try:

        traci.switch(label)
        traci.close()

        print(
            f">>> SUMO closed: {label}",
            flush=True
        )

    except Exception as e:

        print(
            f">>> SUMO close error: {e}",
            flush=True
        )

    time.sleep(0.5)


def test():

    print(
        ">>> Loading DQN model...",
        flush=True
    )

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print(
        f">>> Model loaded: {MODEL_PATH}",
        flush=True
    )

    epsilon = 0.0

    env = TestEnvironment()
    label = "test_run"

    print()
    print("=" * 60)
    print("DQN Traffic Light Test")
    print("=" * 60)

    print(
        "\n>>> Starting SUMO...",
        flush=True
    )

    traci.start(
        [
            SUMO_BINARY,
            "-c",
            SUMO_CONFIG
        ],
        label=label
    )

    traci.switch(label)

    print(
        ">>> SUMO connected.",
        flush=True
    )

    try:

        state = env.reset()

        while (
            traci.simulation.getTime()
            < TOTAL_SIM_TIME
        ):

            if np.random.rand() <= epsilon:

                action = np.random.randint(2)

            else:

                q_values = model.predict(
                    state.reshape(1, -1),
                    verbose=0
                )

                action = int(
                    np.argmax(q_values[0])
                )

            state = env.step(action)

        vehicles_still_present = len(
            traci.vehicle.getIDList()
        )

        vehicles_waiting_at_end = (
            env.queue_total_history[-1]
            if env.queue_total_history
            else 0
        )

    finally:

        close_sumo(label)

    if env.total_waiting_time_history:

        total_waiting_time = float(
            env.total_waiting_time_history[-1]
        )

    else:

        total_waiting_time = 0.0

    if env.total_departed > 0:

        average_waiting_time = (
            total_waiting_time
            / env.total_departed
        )

    else:

        average_waiting_time = 0.0

    if env.queue_total_history:

        average_queue = float(
            np.mean(
                env.queue_total_history
            )
        )

        max_queue = float(
            np.max(
                env.queue_total_history
            )
        )

    else:

        average_queue = 0.0
        max_queue = 0.0

    if env.completed_travel_times:

        average_travel_time = float(
            np.mean(
                env.completed_travel_times
            )
        )

    else:

        average_travel_time = 0.0

    throughput = (
        env.total_arrived
        / TOTAL_SIM_TIME
        * 3600.0
        if TOTAL_SIM_TIME > 0
        else 0.0
    )

    print()
    print("=" * 60)
    print("DQN Test Results")
    print("=" * 60)

    print(
        f"Total waiting time (veh-s): "
        f"{total_waiting_time:.1f}"
    )

    print(
        f"Average waiting time (s): "
        f"{average_waiting_time:.2f}"
    )

    print(
        f"Average queue length (veh): "
        f"{average_queue:.2f}"
    )

    print(
        f"Maximum queue length (veh): "
        f"{max_queue:.0f}"
    )

    print(
        f"Average travel time (s): "
        f"{average_travel_time:.2f}"
    )

    print(
        f"Vehicles generated: "
        f"{env.total_departed}"
    )

    print(
        f"Vehicles arrived: "
        f"{env.total_arrived}"
    )

    print(
        f"Vehicles remaining: "
        f"{vehicles_still_present}"
    )

    print(
        f"Queue at the end: "
        f"{vehicles_waiting_at_end}"
    )

    print(
        f"Final throughput (veh/h): "
        f"{throughput:.2f}"
    )

    print(
        f"Total number of stops: "
        f"{env.total_stops}"
    )

    print("=" * 60)

    with open(
        OUTPUT_CSV,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "time_s",
            "queue_group1",
            "queue_group2",
            "queue_total",
            "vehicles_waiting",
            "vehicles_present",
            "phase",
            "total_waiting_time",
            "vehicles_arrived",
            "vehicles_generated",
            "throughput",
            "avg_waiting_time",
            "number_of_stops"
        ])

        for i in range(len(env.time_history)):

            writer.writerow([
                env.time_history[i],
                env.queue_g1_history[i],
                env.queue_g2_history[i],
                env.queue_total_history[i],
                env.waiting_count_history[i],
                env.vehicles_present_history[i],
                env.phase_history[i],
                env.total_waiting_time_history[i],
                env.arrived_history[i],
                env.departed_history[i],
                env.throughput_history[i],
                env.avg_waiting_time_history[i],
                env.stops_history[i]
            ])

    print(
        "\nCSV saved to:"
    )

    print(OUTPUT_CSV)


if __name__ == "__main__":
    test()
