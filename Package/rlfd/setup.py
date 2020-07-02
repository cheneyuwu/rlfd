import setuptools

setuptools.setup(
    name="rlfd",
    version="0.0.1",
    author="Yuchen Wu",
    author_email="cheney.wu@mail.utoronto.ca",
    description="RL from Demonstrations",
    packages=setuptools.find_packages(),
    license="MIT",
    python_requires='>=3.6',
    install_requires=[
        "tensorflow>=2.2.0", "tensorflow-probability", "matplotlib"
    ],
    zip_safe=False,
)
