import click
from yw.util.plot_util import load_results, plot_results


def launch(dirs, smooth, **kwargs):
    results = load_results(dirs)
    plot_results(results, dirs)


# =============================================================================
# Top Level User API
# =============================================================================
@click.command()
@click.option(
    "--dirs", type=str, default="/home/yuchen/Desktop/FlorianResearch/RLProject/Result", help="Log directory."
)
@click.option("--smooth", type=bool, default="True", help="Smooth the curve.")
def main(**kwargs):
    launch(**kwargs)


if __name__ == "__main__":
    main()
