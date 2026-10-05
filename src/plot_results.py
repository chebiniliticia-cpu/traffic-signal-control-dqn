```python
import pandas as pd
import matplotlib.pyplot as plt
import os

# =========================
# FILES
# =========================

dqn_file = "../data/test_dqn_timeseries_v2.csv"
fixed_file = "../data/test_fixed_timeseries_v2.csv"

dqn = pd.read_csv(dqn_file)
fixed = pd.read_csv(fixed_file)

# Output directory
output_dir = "../figures"
os.makedirs(output_dir, exist_ok=True)


# ============================================================
# 1. WAITING VEHICLES
# ============================================================

plt.figure(figsize=(10, 6), dpi=300)

plt.plot(
    dqn["time_s"],
    dqn["vehicles_waiting"],
    label="DQN"
)

plt.plot(
    fixed["time_s"],
    fixed["vehicles_waiting"],
    label="Fixed-Time"
)

plt.xlabel("Time (s)")
plt.ylabel("Waiting Vehicles")
plt.title("Number of Waiting Vehicles: DQN vs Fixed-Time")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(output_dir, "waiting_vehicles_comparison.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 2. CUMULATIVE WAITING TIME
# ============================================================

plt.figure(figsize=(10, 6), dpi=300)

plt.plot(
    dqn["time_s"],
    dqn["total_waiting_time"],
    label="DQN"
)

plt.plot(
    fixed["time_s"],
    fixed["total_waiting_time"],
    label="Fixed-Time"
)

plt.xlabel("Time (s)")
plt.ylabel("Cumulative Waiting Time (vehicle·s)")
plt.title("Cumulative Waiting Time: DQN vs Fixed-Time")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(output_dir, "cumulative_waiting_time_comparison.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 3. QUEUE LENGTH BY TRAFFIC GROUP
# ============================================================

plt.figure(figsize=(10, 6), dpi=300)

# DQN
plt.plot(
    dqn["time_s"],
    dqn["queue_group1"],
    label="DQN - Group 1"
)

plt.plot(
    dqn["time_s"],
    dqn["queue_group2"],
    label="DQN - Group 2"
)

# Fixed-Time
plt.plot(
    fixed["time_s"],
    fixed["queue_group1"],
    label="Fixed-Time - Group 1",
    linestyle="--"
)

plt.plot(
    fixed["time_s"],
    fixed["queue_group2"],
    label="Fixed-Time - Group 2",
    linestyle="--"
)

plt.xlabel("Time (s)")
plt.ylabel("Queue Length (vehicles)")
plt.title("Queue Length by Traffic Group")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(output_dir, "queue_by_traffic_group.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
```

