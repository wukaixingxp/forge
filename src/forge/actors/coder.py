# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import subprocess
import tempfile
from pathlib import Path

from monarch.actor import endpoint

from forge.controller import ForgeActor

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class SandboxedPythonCoder(ForgeActor):
    """A sandboxed code execution environment using podman containers.

    This actor provides a sandboxed environment for executing Python code
    using podman container technology.

    It automatically manages the entire container lifecycle including image
    pulling, container creation, and cleanup.

    The actor follows a three-stage workflow:
    1. Image Management: Uses podman to pull images from registries
    2. Container Lifecycle: Creates fresh container instances for isolated execution
    3. Code Execution: Safely runs Python code with proper error handling and output capture

    Dependencies:
    - podman: Container engine for pulling images and running containers (must be installed on host)
    - Container images: Accessible via standard container registries

    Args:
        container_image: Container image name to pull (e.g., "python:3.10").
                        Can be any Docker Hub image or custom registry URL.
        container_name: Unique name for the podman container instance. Used for
                        container lifecycle management (create/remove operations).

    """

    def __init__(
        self,
        container_image: str = "python:3.10",
        container_name: str = "sandbox",
    ):
        self.container_image = container_image
        self.container_name = container_name
        self._initialized = False

    @endpoint
    async def setup(self):
        logging.debug("Setting up sandboxed actor")
        await self._maybe_create_image()
        self._recreate()

    @endpoint
    async def recreate(self):
        """Recreates the container instance from the base image."""
        self._recreate()

    async def _maybe_create_image(self):
        """Ensure the container image is pulled and available locally."""
        logging.debug(f"Checking if image {self.container_image} is available")
        
        # Check if image already exists locally
        inspect_result = subprocess.run(
            ["podman", "image", "exists", self.container_image],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        
        if inspect_result.returncode != 0:
            logging.debug(f"Image {self.container_image} not found locally, pulling")
            pull_result = subprocess.run(
                ["podman", "pull", self.container_image],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if pull_result.returncode != 0:
                raise RuntimeError(f"Failed to pull image with podman: {pull_result.stderr}")
            logging.debug(f"Successfully pulled {self.container_image}")
        else:
            logging.info(f"Using existing image: {self.container_image}")

    def _recreate(self):
        """(Re)create a clean container instance from the base image."""
        # Remove any old container
        logging.debug(f"Removing container {self.container_name}")
        subprocess.run(
            ["podman", "rm", "-f", self.container_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        
        # Create new container from image
        # We create the container in a stopped state, ready to be started
        result = subprocess.run(
            [
                "podman", "create",
                "--name", self.container_name,
                "--rm=false",  # We'll manage removal ourselves
                self.container_image,
                "sleep", "infinity"  # Keep container alive for exec commands
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        logging.debug(f"Container creation result: {result}")
        if result.returncode != 0:
            raise RuntimeError(f"Failed to recreate container: {result.stderr}")
        
        # Start the container
        start_result = subprocess.run(
            ["podman", "start", self.container_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if start_result.returncode != 0:
            raise RuntimeError(f"Failed to start container: {start_result.stderr}")
        
        self._initialized = True
        logging.debug("Successfully initialized container")

    @endpoint
    async def execute(self, code: str) -> tuple[str, str]:
        """Executes Python code inside the container and returns the output.

        Args:
            code: Python source code string to execute.

        Returns:
            The captured stdout and stderr from the execution, as a
            (stdout, stderr) tuple of strings.
        """
        logging.debug(f"Executing {code}")
        if not self._initialized:
            raise RuntimeError("Container not initialized. Call recreate() first.")

        # Write code to a temporary file and copy it into the container
        with tempfile.TemporaryDirectory() as tmpdir:
            code_path = Path(tmpdir) / "script.py"
            code_path.write_text(code)

            # Copy the script into the container
            copy_result = subprocess.run(
                [
                    "podman", "cp",
                    str(code_path),
                    f"{self.container_name}:/tmp/script.py"
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if copy_result.returncode != 0:
                raise RuntimeError(f"Failed to copy script to container: {copy_result.stderr}")

            # Execute the code inside the container
            cmd = [
                "podman", "exec",
                self.container_name,
                "python3",
                "/tmp/script.py",
            ]
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            output = result.stdout
            error = result.stderr
            return output, error
