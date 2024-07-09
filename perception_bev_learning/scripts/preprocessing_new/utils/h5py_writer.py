import numpy as np
import h5py


class DatasetWriter:
    def __init__(self, file_path, open_file=None):
        if open_file is not None:
            self.f = open_file
        else:
            self.f = h5py.File(file_path, "w")
        self.compression = "lzf"

    def close(self):
        self.f.close()

    def add_data(self, sequence, fieldname, data_dict, current=None, total=None):
        if sequence not in self.f.keys():
            self.f.create_group(sequence)

        if fieldname not in self.f[sequence]:
            self.f[sequence].create_group(fieldname)

        self.add_dict_to_h5py(self.f[sequence][fieldname], data_dict)

    def add_pointcloud(self, sequence, fieldname, data_dict, current=None, total=None):
        if sequence not in self.f.keys():
            self.f.create_group(sequence)

        if fieldname not in self.f[sequence]:
            self.f[sequence].create_group(fieldname)

        self.add_pointcloud_to_h5py(self.f[sequence][fieldname], data_dict)

    def check_if_exists(self, sequence, fieldname):
        try:
            return fieldname in self.f[sequence]
        except:
            return False

    def add_static(self, sequence, fieldname, data_dict):
        if sequence not in self.f.keys():
            self.f.create_group(sequence)

        if fieldname not in self.f[sequence]:
            self.f[sequence].create_group(fieldname)

        self.add_static_dict_to_h5py(self.f[sequence][fieldname], data_dict)

    def add_static_dict_to_h5py(
        self, group, data_dict, str_list_max_length=50, str_max_length=50
    ):
        for k, v in data_dict.items():
            if k in group.keys():
                continue
            comp = True
            # Convert data element to array
            if isinstance(v, int):
                v = np.array([v])
            elif isinstance(v, float):
                v = np.array([v], dtype=np.float32)
            elif isinstance(v, tuple):
                if isinstance(v[0], str) and isinstance(v[1], int):
                    utf8_type = h5py.string_dtype("utf-8", v[1])
                    v = np.array(v[0].encode("utf-8"), dtype=utf8_type)
            elif isinstance(v, str):
                utf8_type = h5py.string_dtype("utf-8", str_max_length)
                v = np.array(v.encode("utf-8"), dtype=utf8_type)
                comp = False
            elif isinstance(v, list):
                if isinstance(v[0], str):
                    utf8_type = h5py.string_dtype("utf-8", str_list_max_length)
                    v = np.array([st.encode("utf-8") for st in v], dtype=utf8_type)
                else:
                    raise ValueError("not defined")

            if type(v) == np.ndarray:
                # Create new dataset entry
                if comp:
                    group.create_dataset(k, data=v, compression=self.compression)
                else:
                    group.create_dataset(k, data=v)

            else:
                raise ValueError("Error")

    def add_dict_to_h5py(self, group, data_dict):
        for k, v in data_dict.items():
            # Convert data element to array
            if isinstance(v, int):
                v = np.array([v])
            elif isinstance(v, float):
                v = np.array([v], dtype=np.float32)
            elif isinstance(v, tuple):
                if isinstance(v[0], str) and isinstance(v[1], int):
                    utf8_type = h5py.string_dtype("utf-8", v[1])
                    v = np.array(v[0].encode("utf-8"), dtype=utf8_type)

            assert type(v) == np.ndarray

            if k in group.keys():
                # Append to existing dataset
                group[k].resize((group[k].shape[0] + 1), axis=0)
                group[k][-1] = v
            else:
                # Create new dataset entry
                maxshape = (None,) + tuple(v.shape)
                group.create_dataset(
                    k, data=v[None], maxshape=maxshape, compression=self.compression
                )

    def add_pointcloud_to_h5py(self, group, data_dict):
        for k, v in data_dict.items():
            # Convert data element to array
            assert type(v) == np.ndarray

            if k in group.keys():
                # Append to existing dataset
                group[k].resize((group[k].shape[0] + 1), axis=0)
                if group[k].shape[1] < v.shape[0]:
                    group[k].resize((v.shape[0]), axis=1)

                group[k][-1, : v.shape[0]] = v
            else:
                # Create new dataset entry
                maxshape = (None, None)
                group.create_dataset(
                    k, data=v[None], maxshape=maxshape, compression=self.compression
                )
