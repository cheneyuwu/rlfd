import os
import sys

import numpy as np

def main(load_files, store_dir, **kwargs):
    result = None
    for dir in load_files:
        res = {**np.load(dir)}
        if result == None:
            result = res
        else:
            for k in result.keys():
                result[k] = np.concatenate((result[k], res[k]), axis=0)
    
    # store demonstration data
    os.makedirs(store_dir, exist_ok=True)
    file_name = os.path.join(store_dir, "merged_demo.npz")

    np.savez_compressed(file_name, **result)  # save the file
    print("Demo file has been stored into {}.".format(file_name))


if __name__ == "__main__":
    from yw.util.cmd_util import ArgParser

    ap = ArgParser()
    ap.parser.add_argument(
        "--load_file",
        help="directions",
        type=str,
        action="append",
        dest="load_files",
    )
    ap.parser.add_argument(
        "--store_dir",
        help="policy store directory",
        type=str,
        default=os.getenv("PROJECT") + "/Temp/generate_demo/fake_data.npz",
    )
    ap.parse(sys.argv)
    main(**ap.get_dict())
