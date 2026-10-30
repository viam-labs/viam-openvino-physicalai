"""Module entrypoint: importing the service registers its model (EasyResource)."""

import asyncio
import sys

from loguru import logger
from viam.module.module import Module

from src.policy import PhysicalAIPolicy  # noqa: F401 - import registers the model

if __name__ == "__main__":
    # physicalai logs through loguru to stderr, which viam-server shows at error level.
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    asyncio.run(Module.run_from_registry())
