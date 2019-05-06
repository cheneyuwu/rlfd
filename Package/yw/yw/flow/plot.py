import click
import os
from yw.util.plot_util import load_results, plot_results

def plot(dirs, smooth, **kwargs):
    results = load_results(dirs)
    plot_results(results, dirs)


# Top Level User API
# =====================================
@click.command()
@click.option(
    "--dirs", type=str, default=os.getenv("PROJECT")+"/Temp"
)
@click.option("--smooth", type=bool, default="True", help="Smooth the curve.")
def main(**kwargs):
    plot(**kwargs)


if __name__ == "__main__":
    main()