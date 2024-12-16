import os
import pickle
from collections import defaultdict

cg_stat_path = "data/CallGraph/training_data/cg_stats/"

merged_p90_stats = defaultdict(lambda: (0, -1, 0))
# Open all files with extension ".p90" and read the pickle file
for filename in os.listdir(cg_stat_path):
    if filename.endswith(".p90"):
        file_path = os.path.join(cg_stat_path, filename)
        with open(file_path, "rb") as file:
            data = pickle.load(file)

            for svc_id, p90_stats in data.items():
                global_max_lat, global_p90, global_avg = merged_p90_stats[svc_id]
                max_lat, p90, avg = p90_stats
                if max_lat > global_max_lat:
                    global_max_lat = max_lat
                if p90 < global_p90 or global_p90 == -1:
                    global_p90 = p90
                merged_p90_stats[svc_id] = (global_max_lat, global_p90, global_avg)

# Save the merged data to a new file
with open(f"{cg_stat_path}merged_p90_stats.p90", "wb") as file:
    pickle.dump(dict(merged_p90_stats), file)

merged_rare_comm_events = defaultdict(list)
# Open all files with extension ".rare_comm" and read the pickle file
for filename in os.listdir(cg_stat_path):
    if filename.endswith(".rare_comm"):
        file_path = os.path.join(cg_stat_path, filename)
        with open(file_path, "rb") as file:
            data = pickle.load(file)

            for svc_id, rare_comm_events in data.items():
                merged_rare_comm_events[svc_id].extend(rare_comm_events)

# Save the merged data to a new file
with open(f"{cg_stat_path}merged_rare_comm_events.rare_comm", "wb") as file:
    pickle.dump(dict(merged_rare_comm_events), file)
