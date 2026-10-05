import traci
import pandas as pd


SUMO_BINARY = "sumo-gui"

SUMO_CONFIG = (
    r"C:\Users\ASUS\Desktop\traffic_dql"
    r"\traffic_dql.sumocfg"
)

TLS_ID = "319612951"

SIMULATION_TIME = 1800
SEED = 43

GREEN_TIME = 30
YELLOW_TIME = 4

TIMESERIES_FILE = (
    r"C:\Users\ASUS\Desktop\traffic_dql"
    r"\test_fixed_timeseries_v2.csv"
)


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


def get_queue(lanes):

    total = 0

    for lane in lanes:
        try:
            total += traci.lane.getLastStepHaltingNumber(lane)
        except:
            pass

    return total


print("=" * 60)
print("Fixed-Time Traffic Light Test")
print("=" * 60)

sumo_cmd = [
    SUMO_BINARY,
    "-c",
    SUMO_CONFIG,
    "--seed",
    str(SEED)
]

print("\nStarting SUMO...")
traci.start(sumo_cmd)

print("SUMO connected.")


phase = 0
green_elapsed = 0
yellow_remaining = 0

traci.trafficlight.setRedYellowGreenState(
    TLS_ID,
    TLS_GREEN[phase]
)


history = []

sum_queue = 0
max_queue = 0
queue_samples = 0

total_waiting_vehicle_seconds = 0

total_arrived = 0
total_departed = 0

previous_speed = {}
total_stops = 0


print("\nStarting evaluation...\n")


while traci.simulation.getTime() < SIMULATION_TIME:

    traci.simulationStep()

    current_time = traci.simulation.getTime()

    # Vehicles entering the simulation
    total_departed += len(
        traci.simulation.getDepartedIDList()
    )

    # Queue lengths
    q1 = get_queue(LANES_GROUP_1)
    q2 = get_queue(LANES_GROUP_2)

    total_queue = q1 + q2

    # Vehicles currently in the simulation
    present_ids = traci.vehicle.getIDList()
    vehicles_present = len(present_ids)

    waiting_vehicles = 0

    for veh_id in present_ids:

        try:
            speed = traci.vehicle.getSpeed(veh_id)

            if speed < 0.1:
                waiting_vehicles += 1

            previous = previous_speed.get(
                veh_id,
                100.0
            )

            if previous >= 0.1 and speed < 0.1:
                total_stops += 1

            previous_speed[veh_id] = speed

        except:
            pass

    # Vehicles that reached their destination
    arrived_now = traci.simulation.getArrivedNumber()
    total_arrived += arrived_now

    for veh_id in traci.simulation.getArrivedIDList():
        previous_speed.pop(veh_id, None)

    # One second of waiting time for each waiting vehicle
    total_waiting_vehicle_seconds += waiting_vehicles

    sum_queue += total_queue
    queue_samples += 1

    if total_queue > max_queue:
        max_queue = total_queue

    if total_departed > 0:
        average_waiting_time = (
            total_waiting_vehicle_seconds
            / total_departed
        )
    else:
        average_waiting_time = 0.0

    if current_time > 0:
        throughput = (
            total_arrived
            / current_time
            * 3600.0
        )
    else:
        throughput = 0.0

    history.append({
        "time_s": current_time,
        "queue_group1": q1,
        "queue_group2": q2,
        "queue_total": total_queue,
        "vehicles_waiting": waiting_vehicles,
        "vehicles_present": vehicles_present,
        "phase": phase,
        "total_waiting_time": total_waiting_vehicle_seconds,
        "vehicles_arrived": total_arrived,
        "vehicles_generated": total_departed,
        "throughput": throughput,
        "avg_waiting_time": average_waiting_time,
        "number_of_stops": total_stops
    })

    # Fixed-time traffic light
    if yellow_remaining > 0:

        yellow_remaining -= 1

        if yellow_remaining == 0:

            phase = 1 - phase

            traci.trafficlight.setRedYellowGreenState(
                TLS_ID,
                TLS_GREEN[phase]
            )

            green_elapsed = 0

    else:

        green_elapsed += 1

        if green_elapsed >= GREEN_TIME:

            traci.trafficlight.setRedYellowGreenState(
                TLS_ID,
                TLS_YELLOW[phase]
            )

            yellow_remaining = YELLOW_TIME

    if int(current_time) % 100 == 0:

        print(
            f"Time: {int(current_time)} / "
            f"{SIMULATION_TIME}s | "
            f"Phase: {phase} | "
            f"Queue: {total_queue}"
        )


print("\nSimulation finished.")


if history:

    vehicles_at_end = history[-1]["vehicles_present"]
    waiting_at_end = history[-1]["vehicles_waiting"]

else:

    vehicles_at_end = 0
    waiting_at_end = 0


if queue_samples > 0:

    average_queue = sum_queue / queue_samples

else:

    average_queue = 0.0


total_waiting = total_waiting_vehicle_seconds


if total_departed > 0:

    average_waiting = (
        total_waiting
        / total_departed
    )

else:

    average_waiting = 0.0


throughput_final = (
    total_arrived
    / SIMULATION_TIME
    * 3600.0
    if SIMULATION_TIME > 0
    else 0.0
)


df = pd.DataFrame(history)

df.to_csv(
    TIMESERIES_FILE,
    index=False
)


print("\n" + "=" * 60)
print("Fixed-Time Test Results")
print("=" * 60)

print(
    f"Total waiting time (veh-s): "
    f"{total_waiting:.1f}"
)

print(
    f"Average waiting time (s): "
    f"{average_waiting:.2f}"
)

print(
    f"Average queue length (veh): "
    f"{average_queue:.2f}"
)

print(
    f"Maximum queue length (veh): "
    f"{max_queue}"
)

print(
    f"Vehicles generated: "
    f"{total_departed}"
)

print(
    f"Vehicles arrived: "
    f"{total_arrived}"
)

print(
    f"Vehicles remaining: "
    f"{vehicles_at_end}"
)

print(
    f"Vehicles waiting at the end: "
    f"{waiting_at_end}"
)

print(
    f"Final throughput (veh/h): "
    f"{throughput_final:.2f}"
)

print(
    f"Total number of stops: "
    f"{total_stops}"
)

print("=" * 60)

print(
    "\nTime series saved to:"
)

print(TIMESERIES_FILE)


print("\nClosing SUMO...")

try:
    traci.close()
except:
    pass

print("SUMO closed.")
