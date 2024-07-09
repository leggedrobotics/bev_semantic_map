import numpy as np

PDC_DATATYPE = {
    "1": np.int8,
    "2": np.uint8,
    "3": np.int16,
    "4": np.uint16,
    "5": np.int32,
    "6": np.uint32,
    "7": np.float32,
    "8": np.float64,
}
COUNTER_DIGITS = 10
SECONDS_DIGITS = 10
NSECONDS_DIGITS = 9
IMAGE_OUTPUT_FORMAT = ".png"


def counter_to_str(counter):
    counterStr = str(counter)
    return counterStr.zfill(COUNTER_DIGITS)


def secs_to_str(secs):
    secsStr = str(secs)
    return secsStr.zfill(SECONDS_DIGITS)


def nsecs_to_str(nsecs):
    nsecsStr = str(nsecs)
    return nsecsStr.zfill(NSECONDS_DIGITS)
