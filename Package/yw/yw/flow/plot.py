import os
import sys
from yw.util.plot_util import load_results, plot_results

def plot(dirs, smooth, **kwargs):
    results = load_results(dirs)
    plot_results(results, dirs)


if __name__ == "__main__":
    from yw.util.cmd_util import ArgParser
    ap = ArgParser()
    # logging
    ap.parser.add_argument("--dirs", help="load directory", type=str, default=os.getenv("PROJECT")+"/Temp")
    ap.parser.add_argument("--smooth", help="smooth the curve", type=int, default=1)
    ap.parse(sys.argv)
    plot(**ap.get_dict())