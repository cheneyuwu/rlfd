from setuptools import setup

setup(
    name="YWRL",
    version="0.1",
    description="Yuchen's implementation of RL algorithms",
    # url='http://github.com/storborg/funniest',
    author="Yuchen Wu",
    author_email="cheney.wu@mail.utoronto.ca",
    license="MIT",
    packages=["yw"],
    install_requires=[
        "gym",
        "matplotlib",
        "mpi4py",
        # "tensorflow==1.13.1",
        "tensorflow_probability",
        "pandas",
    ],
    zip_safe=False,
)

