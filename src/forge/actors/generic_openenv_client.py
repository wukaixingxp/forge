# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Generic OpenEnv Actor using GenericEnvClient.

This actor works with ANY OpenEnv environment using only raw dictionaries,
without requiring environment-specific packages (like julia_env or coding_env).

Usage:
    from openenv import GenericEnvClient, GenericAction

    # Create actor for any environment - just specify the Docker image
    actor = GenericOpenEnvClientActor(
        docker_image="julia-env:latest",
    )
    await actor.setup()

    # Execute with GenericAction (just a dict wrapper)
    action = GenericAction(core_code="println('hello')", test_code="@test true")
    result = await actor.execute(action)  # Returns StepResult with dict observation

    await actor.teardown()
"""

import logging
import socket
import traceback
from typing import Any, Dict, Optional

from openenv import GenericEnvClient, GenericAction
from openenv.core.client_types import StepResult
from monarch.actor import endpoint

from forge.controller import ForgeActor
from forge.observability.metrics import record_metric, Reduce

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """Check if a port is already in use on the specified host."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return False
        except OSError:
            return True


def find_available_port(
    preferred_port: int, min_port: int = 5000, max_attempts: int = 100
) -> int:
    """Find an available port starting from preferred_port and decrementing."""
    port = preferred_port
    attempts = 0

    while attempts < max_attempts:
        if port < min_port:
            raise RuntimeError(
                f"No available port found after trying {attempts} ports. "
                f"Reached minimum port {min_port}."
            )

        if not is_port_in_use(port):
            logger.info(f"Found available port: {port}")
            return port

        logger.debug(f"Port {port} is in use, trying {port - 1}")
        port -= 1
        attempts += 1

    raise RuntimeError(
        f"No available port found after trying {max_attempts} ports "
        f"(from {preferred_port} down to {port})."
    )


class GenericOpenEnvClientActor(ForgeActor):
    """A generic sandboxed execution environment using GenericEnvClient.

    This actor uses OpenEnv's GenericEnvClient to work with ANY OpenEnv environment
    using only raw dictionaries (Dict[str, Any]). No environment-specific packages
    required - just specify the Docker image.

    Benefits over GenericOpenEnvActor:
    - No need to import environment-specific packages (julia_env, coding_env, etc.)
    - Works with any OpenEnv-compatible Docker image
    - Type-safe with GenericAction dictionary wrapper
    - Lower coupling - doesn't require AutoEnv/AutoAction discovery

    Usage:
        >>> actor = GenericOpenEnvClientActor(docker_image="julia-env:latest")
        >>> await actor.setup()
        >>> action = GenericAction(core_code="...", test_code="...")
        >>> result = await actor.execute(action)
        >>> print(result.observation)  # Dict with exit_code, stdout, stderr, etc.
        >>> await actor.teardown()
    """

    def __init__(
        self,
        docker_image: str,
        env_vars: Optional[Dict[str, str]] = None,
        container_timeout_s: float = 180.0,
        request_timeout_s: float = 120.0,
        port: int = 8000,
        container_memory_gb: int = 4,
        enable_zombie_cleanup: bool = False,
    ):
        """Initialize the generic OpenEnv actor.

        Args:
            docker_image: Docker image name (e.g., "julia-env:latest", "coding-env:latest")
            env_vars: Environment variables to pass to the container
            container_timeout_s: Timeout for container startup in seconds
            request_timeout_s: Timeout for individual requests in seconds
            port: Preferred port for the container
            container_memory_gb: Memory limit for the container in GB
            enable_zombie_cleanup: Whether to enable zombie process cleanup
        """
        self.docker_image = docker_image
        self.env_vars = env_vars or {}
        self.container_timeout_s = container_timeout_s
        self.request_timeout_s = request_timeout_s
        self.port = port
        self.container_memory_gb = container_memory_gb
        self.enable_zombie_cleanup = enable_zombie_cleanup
        self.client: Optional[GenericEnvClient] = None

    @endpoint
    async def setup(self):
        """Initialize the OpenEnv environment and start the container."""
        logging.debug(
            f"Setting up GenericEnvClient actor with image {self.docker_image}"
        )
        try:
            # Find an available port
            available_port = find_available_port(self.port)
            if available_port != self.port:
                logger.warning(
                    f"Preferred port {self.port} is in use. Using port {available_port} instead."
                )
            else:
                logger.info(f"Using port {available_port}")

            # Update PORT env var
            self.env_vars["PORT"] = str(available_port)

            if self.env_vars:
                logger.debug(
                    f"Passing environment variables to container: {self.env_vars}"
                )

            # Create GenericEnvClient using from_docker_image
            self.client = GenericEnvClient.from_docker_image(
                self.docker_image,
                wait_timeout=self.container_timeout_s,
                request_timeout_s=self.request_timeout_s,
                env_vars=self.env_vars,
                port=available_port,
                memory_gb=self.container_memory_gb,
            )
            logging.debug("Successfully initialized GenericEnvClient.")

            if self.client:
                self.client.reset()
                logging.debug("Initial environment reset complete.")

        except TimeoutError as e:
            logging.error(
                f"Container failed to start within timeout: {e}\n"
                "Please check Docker logs for more details."
            )
            raise
        except Exception as e:
            logging.error(f"Failed to setup GenericEnvClient: {e}")
            raise

    @endpoint
    async def recreate(self):
        """Resets the environment to a clean state."""
        if not self.client:
            raise RuntimeError("Client not initialized. Call setup() first.")
        logging.debug("Recreating environment state (resetting).")
        self.client.reset()
        logging.debug("Environment reset.")

    @endpoint
    async def execute(self, action: Dict[str, Any]) -> StepResult[Dict[str, Any]]:
        """Executes an action inside the environment and returns the result.

        Args:
            action: Dictionary action (or GenericAction). For Julia environments,
                   this should contain {"core_code": "...", "test_code": "..."}

        Returns:
            StepResult containing:
                - observation: Dict with keys like exit_code, stdout, stderr,
                              tests_passed, tests_failed, code_compiles, etc.
                - reward: Float reward value (if environment provides it)
                - done: Boolean indicating if episode is complete
        """
        logging.debug(f"Executing action: {action}")
        if not self.client:
            raise RuntimeError("Client not initialized. Call setup() first.")

        max_retries = 2
        for attempt in range(max_retries):
            try:
                result = self.client.step(action)
                return result

            except Exception as e:
                error_msg = str(e).lower()
                is_container_error = any(
                    keyword in error_msg
                    for keyword in [
                        "no such container",
                        "container not found",
                        "container is not running",
                        "container has stopped",
                        "connection refused",
                    ]
                )

                if is_container_error:
                    record_metric("container/error_count", 1, Reduce.SUM)
                    logging.error(
                        f"Container error on attempt {attempt + 1}/{max_retries}: {e}"
                    )

                    if attempt < max_retries - 1:
                        logging.info("Attempting to recreate environment...")
                        try:
                            self.client.close()
                            import time
                            time.sleep(3)

                            available_port = find_available_port(self.port)
                            self.client = GenericEnvClient.from_docker_image(
                                self.docker_image,
                                wait_timeout=self.container_timeout_s,
                                request_timeout_s=self.request_timeout_s,
                                env_vars=self.env_vars,
                                port=available_port,
                                memory_gb=self.container_memory_gb,
                            )
                            self.client.reset()
                            logging.info("Environment recreated successfully")
                            continue
                        except Exception as recreate_error:
                            logging.error(
                                f"Failed to recreate environment: {recreate_error}"
                            )
                            if attempt == max_retries - 1:
                                raise RuntimeError(
                                    f"Container failed and could not be recreated: {e}"
                                ) from e
                    else:
                        raise RuntimeError(
                            f"Container failed after {max_retries} attempts: {e}"
                        ) from e
                else:
                    raise

        raise RuntimeError("Execution failed after all retry attempts")

    @endpoint
    async def get_state(self) -> Dict[str, Any]:
        """Get the current environment state as a dictionary."""
        if not self.client:
            raise RuntimeError("Client not initialized. Call setup() first.")
        return self.client.state()

    @endpoint
    async def teardown(self):
        """Cleans up the environment and stops the container."""
        if self.client:
            logging.debug("Closing GenericEnvClient and stopping container.")
            self.client.close()
            self.client = None
            logging.debug("Cleanup complete.")

    def create_action(self, **kwargs) -> GenericAction:
        """Helper method to create a GenericAction.

        Args:
            **kwargs: Arguments for the action (e.g., core_code, test_code)

        Returns:
            GenericAction instance (dictionary wrapper)

        Example:
            action = actor.create_action(core_code="println(1)", test_code="@test true")
        """
        return GenericAction(**kwargs)
