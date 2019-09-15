from setuptools import setup

setup(
    name="td3fd2",
    version="0.0",
    description="TD3fD through Shaping using Generative Models",
    author="Yuchen Wu",
    author_email="cheney.wu@mail.utoronto.ca",
    license="MIT",
    packages=["td3fd"],
    install_requires=["matplotlib", "tensorflow==1.14.0", "tensorflow_probability==0.7.0", "pandas"],
    zip_safe=False,
)
