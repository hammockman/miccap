"""
A datastore for acquired signals.

jh, Nov 2021
"""

import h5py
import datetime

def make_id_unique(h5, signalid):
    if f"/{signalid}" not in h5:
        return signalid
    import re
    patt = re.compile("(.+)-([0-9]+)$")
    m = patt.search(signalid)
    return make_id_unique(h5, f"{signalid}-1" if m is None else f"{m.group(1)}-{int(m.group(2))+1}")


def write(fn, signalid, signal, meta=None):
    with h5py.File(fn, 'a') as f:
        # todo: catch case where this dataset exists already
        uuid = make_id_unique(f, signalid)
        dset = f.create_dataset(uuid, data=signal)
        dset.attrs['capture_time'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for k,v in meta.items():
            if v is not None:
                dset.attrs[k] = str(v)
    return uuid


if __name__=="__main__":
    import sys
    print(make_id_unique(h5py.File('signals.h5','r'), sys.argv[1]))
