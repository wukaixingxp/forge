# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import sys

from envs.coding_env import CodeAction, CodingEnv

from forge.controller import ForgeActor
from monarch.actor import endpoint

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class OpenEnvCoder(ForgeActor):
    """A sandboxed code execution environment using OpenEnv.

    This actor uses OpenEnv's CodingEnv to provide a sandboxed
    environment for executing Python code.

    It automatically manages the entire container lifecycle including
    image startup, environment connection, and cleanup.

    The actor follows a three-stage workflow:
    1. Environment Initialization: Creates a client from a Docker image.
    2. State Management: Resets the environment for a clean execution state.
    3. Code Execution: Safely runs Python code with proper error handling and output capture.

    Dependencies:
    - Docker: Must be installed and running on the host.
    - OpenEnv: The OpenEnv library must be in the PYTHONPATH.
    - Docker image: A compatible Docker image (e.g., "coding-env:latest").

    Args:
        docker_image: Docker image to use for the environment (e.g., "coding-env:latest").
    """

    def __init__(
        self,
        docker_image: str = "coding-env:latest",
    ):
        self.docker_image = docker_image
        self.client: CodingEnv | None = None

    @endpoint
    async def setup(self):
        logging.debug(f"Setting up OpenEnv actor with image {self.docker_image}")
        self.client = CodingEnv.from_docker_image(self.docker_image)
        logging.debug("Successfully initialized OpenEnv client.")
        if self.client:
            self.client.reset()
            logging.debug("Initial environment reset complete.")

    @endpoint
    async def recreate(self):
        """Resets the environment to a clean state."""
        if not self.client:
            raise RuntimeError("Client not initialized. Call setup() first.")
        logging.debug("Recreating environment state (resetting).")
        self.client.reset()
        logging.debug("Environment reset.")

    @endpoint
    async def execute(self, code: str) -> tuple[str, str]:
        """Executes Python code inside the environment and returns the output.

        Args:
            code: Python source code string to execute.

        Returns:
            The captured stdout and stderr from the execution, as a
            (stdout, stderr) tuple of strings.
        """
        print("=" * 80)
        print("[DEBUG] INPUT CODE:")
        print("-" * 80)
        print(code)
        print("-" * 80)
        logging.debug(f"Executing {code}")
        if not self.client:
            raise RuntimeError("Client not initialized. Call setup() first.")

        result = self.client.step(CodeAction(code=code))

        output = result.observation.stdout
        error = result.observation.stderr

        print("[DEBUG] EXECUTION OUTPUTS:")
        print("-" * 80)
        print(f"Return Code: {result.observation.exit_code}")
        print(f"\nSTDOUT:\n{output if output else '(empty)'}")
        print(f"\nSTDERR:\n{error if error else '(empty)'}")
        print("=" * 80)

        return output, error

    @endpoint
    async def teardown(self):
        """Cleans up the environment and stops the container."""
        if self.client:
            logging.debug("Closing OpenEnv client and stopping container.")
            self.client.close()
            self.client = None
            logging.debug("Cleanup complete.")
