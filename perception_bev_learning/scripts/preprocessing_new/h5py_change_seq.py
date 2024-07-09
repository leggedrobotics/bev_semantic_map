import h5py

filename = "/data/bev_traversability/tensor01/halter_ranch/racer-jpl9_hw_2023-08-29-21-48-16-UTC_halter-ranch-dubost-2_ROBOT_dubost-datacollect-patrick-1.h5py"

old_seq_name = "playback_slow"
new_seq_name = "racer-jpl9_hw_2023-08-29-21-48-16-UTC_halter-ranch-dubost-2_ROBOT_dubost-datacollect-patrick-1"

# Open the HDF5 file in read-write mode
with h5py.File(filename, "r+") as file:
    # Rename the sequence from 'old_sequence_name' to 'new_sequence_name'
    file.move(old_seq_name, new_seq_name)

    # Save the changes
    file.flush()
