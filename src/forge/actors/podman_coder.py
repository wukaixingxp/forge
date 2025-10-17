# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import logging
import subprocess
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from forge.controller import ForgeActor

from monarch.actor import endpoint

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class PodmanPythonCoder(ForgeActor):
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
        max_workers: Maximum number of concurrent subprocess executions (default: 4).

    """

    def __init__(
        self,
        container_image: str = "python:3.10",
        container_name: str = "sandbox",
        max_workers: int = 4,
    ):
        self.container_image = container_image
        self.container_name = container_name
        self._initialized = False
        # Thread pool for running subprocess calls without blocking event loop
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="podman_exec_"
        )

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
                raise RuntimeError(
                    f"Failed to pull image with podman: {pull_result.stderr}"
                )
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
                "podman",
                "create",
                "--name",
                self.container_name,
                "--rm=false",  # We'll manage removal ourselves
                self.container_image,
                "sleep",
                "infinity",  # Keep container alive for exec commands
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

    def _run_subprocess_with_retry(
        self, cmd: list[str], max_retries: int = 3, timeout: int | None = None
    ) -> subprocess.CompletedProcess:
        """Run subprocess with exponential backoff retry on resource exhaustion.

        Args:
            cmd: Command to run as list of strings
            max_retries: Maximum number of retry attempts (default: 3)
            timeout: Timeout in seconds for subprocess (default: None)

        Returns:
            subprocess.CompletedProcess result

        Raises:
            RuntimeError: If all retries fail
        """
        last_error = None

        for attempt in range(max_retries):
            try:
                return subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=timeout,
                )
            except BlockingIOError as e:
                last_error = e
                if attempt < max_retries - 1:
                    # Exponential backoff: 0.1s, 0.2s, 0.4s
                    wait_time = 0.1 * (2**attempt)
                    logger.warning(
                        f"BlockingIOError on attempt {attempt + 1}/{max_retries} for cmd {cmd[0:2]}, "
                        f"retrying in {wait_time}s: {e}"
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(
                        f"BlockingIOError persisted after {max_retries} attempts for cmd {cmd[0:2]}: {e}"
                    )
            except Exception as e:
                logger.error(f"Unexpected error running subprocess {cmd[0:2]}: {e}")
                raise

        # If we get here, all retries failed
        raise RuntimeError(
            f"Failed to run subprocess after {max_retries} attempts. "
            f"Last error: {last_error}. This indicates system resource exhaustion. "
            f"Consider reducing max_workers or increasing system process limits."
        )

    def _execute_sync(self, code: str) -> tuple[str, str]:
        """Synchronous code execution - runs in thread pool via run_in_executor.

        This method contains the actual subprocess calls and is designed to be
        run in a thread pool to avoid blocking the async event loop.

        Uses UUID-based filenames to avoid race conditions when multiple threads
        execute code concurrently in the same container.
        """
        # Generate unique script name to avoid race conditions between concurrent executions
        execution_id = uuid.uuid4().hex[:8]  # Short UUID for readability
        script_name = f"script_{execution_id}.py"
        container_script_path = f"/tmp/{script_name}"

        logging.debug(f"Executing code in thread pool (execution_id={execution_id})")

        # Write code to a temporary file and copy it into the container
        with tempfile.TemporaryDirectory() as tmpdir:
            code_path = Path(tmpdir) / script_name
            code_path.write_text(code)

            # Copy the script into the container with retry logic
            copy_result = self._run_subprocess_with_retry(
                [
                    "podman",
                    "cp",
                    str(code_path),
                    f"{self.container_name}:{container_script_path}",
                ],
                max_retries=3,
            )
            if copy_result.returncode != 0:
                raise RuntimeError(
                    f"Failed to copy script to container: {copy_result.stderr}"
                )

            # Execute the code inside the container with 300 second timeout and retry logic
            try:
                result = self._run_subprocess_with_retry(
                    [
                        "podman",
                        "exec",
                        self.container_name,
                        "python3",
                        container_script_path,
                    ],
                    max_retries=3,
                    timeout=300,
                )
                output = result.stdout
                error = result.stderr
            except subprocess.TimeoutExpired:
                logging.warning(
                    f"Code execution timed out after 300 seconds (execution_id={execution_id})"
                )
                output = ""
                error = "Error: Code execution timed out after 300 seconds (possible infinite loop)"
            finally:
                # Clean up the script file from container to avoid clutter
                # Use check=False to not fail if file doesn't exist
                subprocess.run(
                    [
                        "podman",
                        "exec",
                        self.container_name,
                        "rm",
                        "-f",
                        container_script_path,
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )

            return output, error

    @endpoint
    async def execute(self, code: str) -> tuple[str, str]:
        """Executes Python code inside the container and returns the output.

        Uses ThreadPoolExecutor with run_in_executor to run synchronous subprocess
        calls without blocking the async event loop. This is simpler and more
        reliable than trying to make subprocess async.

        Args:
            code: Python source code string to execute.

        Returns:
            The captured stdout and stderr from the execution, as a
            (stdout, stderr) tuple of strings.
        """
        if not self._initialized:
            raise RuntimeError("Container not initialized. Call recreate() first.")

        # Run synchronous code execution in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self._execute_sync, code)

    @endpoint
    async def shutdown(self):
        """Cleanup resources - shutdown thread pool executor."""
        logging.debug("Shutting down PodmanPythonCoder thread pool")
        try:
            self._executor.shutdown(wait=True, cancel_futures=False)
            logging.info("Thread pool executor shutdown successfully")
        except Exception as e:
            logging.error(f"Error during thread pool shutdown: {e}")
