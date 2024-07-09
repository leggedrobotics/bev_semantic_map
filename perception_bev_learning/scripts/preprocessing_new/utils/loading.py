import pickle


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
