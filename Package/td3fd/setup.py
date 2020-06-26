from setuptools import setup

setup(
    name="td3fd",
    version="0.1",
    description="TD3fD through Shaping using Generative Models",
    author="Yuchen Wu",
    author_email="cheney.wu@mail.utoronto.ca",
    license="MIT",
    packages=["td3fd"],
    install_requires=["matplotlib", "pandas",],
    zip_safe=False,
)
