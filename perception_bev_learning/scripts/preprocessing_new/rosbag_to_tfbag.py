import rosbag
import os
import sys
import argparse
from tqdm import tqdm


def read_and_write_bags(input_bag_files, output_bag_file):
    # Create a new bag file to write the filtered messages
    with rosbag.Bag(output_bag_file, 'w') as outbag:
        for bag_file in input_bag_files:
            with rosbag.Bag(bag_file, 'r') as inbag:
                # Get total number of messages for progress bar
                total_messages = sum(1 for _ in inbag.read_messages(topics=['/tf', '/tf_static']))
                with tqdm(total=total_messages, desc=f"Processing {os.path.basename(bag_file)}", unit="msg") as pbar:
                    # Iterate through the messages in the bag
                    for topic, msg, t in inbag.read_messages(topics=['/tf', '/tf_static']):
                        # Write the message to the new bag file
                        outbag.write(topic, msg, t)
                        pbar.update(1)

def get_bag_files_from_directory(directory):
    # Get all .bag files in the specified directory
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.bag')]

if __name__ == '__main__':
    # Setup command line argument parsing
    parser = argparse.ArgumentParser(description='Filter tf and tf_static messages from multiple bag files into a new bag file.')
    parser.add_argument('directory', type=str, help='Directory containing the input bag files.')
    parser.add_argument('--output_bag', type=str, help='Output bag file to store filtered messages.', default=None)

    args = parser.parse_args()

    # Get list of input bag files from the provided directory
    input_bag_files = get_bag_files_from_directory(args.directory)

    if not input_bag_files:
        print(f'No bag files found in the directory: {args.directory}')
        sys.exit(1)

    # Default output bag file in the same directory as input files
    output_bag_file = args.output_bag if args.output_bag else os.path.join(args.directory, 'bev_tf.bag')

    # Read from input bag files and write to output bag file
    read_and_write_bags(input_bag_files, output_bag_file)

    print(f'Finished writing {output_bag_file}')
