"""Module entrypoint: importing the service registers its model (EasyResource)."""

import asyncio

from viam.module.module import Module

from src.policy import PhysicalAIPolicy  # noqa: F401 - import registers the model

if __name__ == "__main__":
    asyncio.run(Module.run_from_registry())
