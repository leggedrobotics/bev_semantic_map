import numpy as np


class mean:
    def __init__(self) -> None:
        pass

    def fuse(self, measurement, map1, map2, *args, **kwargs):
        m = ~np.isnan(measurement)
        m_invalid = np.isnan(map1)

        map1[m_invalid * m] = measurement[m_invalid * m]
        map1[~m_invalid * m] += measurement[~m_invalid * m]
        map2[m] += 1
        return map1, map2


class latest:
    def __init__(self) -> None:
        pass

    def fuse(self, measurement, map1, map2, *args, **kwargs):
        m = ~np.isnan(measurement)
        map1[m] = measurement[m]
        map2[m] = 1
        return map1, map2


class latest_reliable:
    def __init__(self, threshold=0.5) -> None:
        self.threshold = threshold

    def fuse(self, measurement, map1, map2, reliable, *args, **kwargs):
        meas_valid = ~np.isnan(measurement) * (reliable > self.threshold)
        # measurement valid
        map1[meas_valid] = measurement[meas_valid]
        map2[meas_valid] = 1

        return map1, map2


class maximum:
    def __init__(self) -> None:
        pass

    def fuse(self, measurement, map1, map2, *args, **kwargs):
        meas_valid = ~np.isnan(measurement)

        map_invalid = np.isnan(map1)

        # map cell invalid but measurement valid
        map1[map_invalid * meas_valid] = measurement[map_invalid * meas_valid]

        # both inputs valid and measurement is greater
        m_both_valid = meas_valid * ~map_invalid
        m_measurement_greater = np.nan_to_num(measurement) > np.nan_to_num(map1)
        map1[m_both_valid * m_measurement_greater] = measurement[
            m_both_valid * m_measurement_greater
        ]
        map2[:, :] = 1
        return map1, map2


class maximum_reliable:
    def __init__(self, threshold=0.5) -> None:
        self.threshold = threshold

    def fuse(self, measurement, map1, map2, reliable, *args, **kwargs):
        meas_valid = ~np.isnan(measurement) * (reliable > self.threshold)

        map_invalid = np.isnan(map1)

        # map cell invalid but measurement valid
        map1[map_invalid * meas_valid] = measurement[map_invalid * meas_valid]

        # both inputs valid and measurement is greater
        m_both_valid = meas_valid * ~map_invalid
        m_measurement_greater = np.nan_to_num(measurement) > np.nan_to_num(map1)
        map1[m_both_valid * m_measurement_greater] = measurement[
            m_both_valid * m_measurement_greater
        ]
        map2[:, :] = 1
        return map1, map2


class minimum:
    def __init__(self) -> None:
        pass

    def fuse(self, measurement, map1, map2, *args, **kwargs):
        meas_valid = ~np.isnan(measurement)

        map_invalid = np.isnan(map1)

        # map cell invalid but measurement valid
        map1[map_invalid * meas_valid] = measurement[map_invalid * meas_valid]

        # both inputs valid and measurement is smaller
        m_both_valid = meas_valid * ~map_invalid
        m_measurement_smaller = np.nan_to_num(measurement) < np.nan_to_num(map1)
        map1[m_both_valid * m_measurement_smaller] = measurement[
            m_both_valid * m_measurement_smaller
        ]
        map2[~np.isnan(map1)] = 1
        return map1, map2


class minimum_reliable:
    def __init__(self, threshold=0.5) -> None:
        self.threshold = threshold

    def fuse(self, measurement, map1, map2, reliable, *args, **kwargs):
        meas_valid = ~np.isnan(measurement) * (reliable > self.threshold)

        map_invalid = np.isnan(map1)

        # map cell invalid but measurement valid
        map1[map_invalid * meas_valid] = measurement[map_invalid * meas_valid]

        # both inputs valid and measurement is greater
        m_both_valid = meas_valid * ~map_invalid
        m_measurement_smaller = np.nan_to_num(measurement) < np.nan_to_num(map1)
        map1[m_both_valid * m_measurement_smaller] = measurement[
            m_both_valid * m_measurement_smaller
        ]
        map2[:, :] = 1
        return map1, map2
