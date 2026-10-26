from setuptools import find_packages, setup

# Dependencies live in the custom training container (see ../Dockerfile), so the
# job doesn't reinstall them on billed GPU time.
setup(name="physical-ai-studio", version="0.1.0", packages=find_packages(), install_requires=[])
