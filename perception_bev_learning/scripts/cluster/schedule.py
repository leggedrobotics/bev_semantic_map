import os

# Execute this script either on GE76 or cluster to correctly start the job
# Only provide exp yaml.


# Read experiment files

system = os.environ.get("ENV_WORKSTATION_NAME")

if system == "ge76":
    # 1. Sync to cluster
    pass
    # 2. Remotely schedule experiments via ssh


if system == "tacc-lonestar6":
    # 1. Submit all jobs
    pass
