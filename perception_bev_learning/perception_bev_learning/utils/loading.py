import os
import yaml


import pickle
from perception_bev_learning import BEV_ROOT_DIR

__all__ = ["file_path", "load_yaml", "load_pkl", "load", "dump"]


def file_path(string: str) -> str:
    """Checks if string is a file path
    Args:
        string (str): Potential file path
    Raises:
        NotADirectoryError: String is not a fail path
    Returns:
        (str): Returns the file path
    """

    if os.path.isfile(string):
        return string
    else:
        raise NotADirectoryError(string)


def load_yaml(path: str) -> dict:
    """Loads yaml file
    Args:
        path (str): File path
    Returns:
        (dict): Returns content of file
    """
    with open(path, "rb") as file:
        res = yaml.load(file, Loader=yaml.FullLoader)
    return res


def load_pkl(path: str) -> dict:
    """Loads pkl file
    Args:
        path (str): File path
    Returns:
        (dict): Returns content of file
    """
    with open(path, "rb") as file:
        res = pickle.load(file)
    return res


def dump(data: dict):
    """Stores to pkl file
    Args:
        data (dict):
    """
    with open(os.path.join(os.path.expanduser("~"), "pickle_dump.pkl"), "wb") as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)


def load() -> dict:
    """Loads pkl file
    Returns:
        (dict): Returns content of file
    """
    with open(os.path.join(os.path.expanduser("~"), "pickle_dump.pkl"), "rb") as file:
        return pickle.load(file)


def load_env() -> dict:
    """Uses ENV_WORKSTATION_NAME variable to load specified environment yaml file.

    Returns:
        (dict): Returns content of environment file
    """
    env_cfg_path = os.path.join(BEV_ROOT_DIR, "cfg/env", os.environ["ENV_WORKSTATION_NAME"] + ".yaml")
    env = load_yaml(env_cfg_path)
    for k in env.keys():
        if k == "workstation":
            continue
        if not os.path.isabs(env[k]):
            env[k] = os.path.join(BEV_ROOT_DIR, env[k])

    return env
