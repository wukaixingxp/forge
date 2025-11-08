# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import socket
import traceback
from typing import Any, Dict, Generic, Optional, Type, TypeVar

from core.client_types import StepResult
from core.env_server.types import Action, Observation
from core.http_env_client import HTTPEnvClient
from monarch.actor import endpoint

from forge.controller import ForgeActor
from forge.observability.metrics import record_metric, Reduce

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Type variables for generic action and observation types
ActT = TypeVar("ActT", bound=Action)
ObsT = TypeVar("ObsT", bound=Observation)


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """
    Check if a port is already in use on the specified host.

    Args:
        port: Port number to check
        host: Host address to check (default: 127.0.0.1)

    Returns:
        True if port is in use, False if available
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            # Try to bind to the port
            s.bind((host, port))
            return False  # Port is available
        except OSError:
            return True  # Port is in use


def find_available_port(
    preferred_port: int, min_port: int = 5000, max_attempts: int = 100
) -> int:
    """
    Find an available port starting from preferred_port and decrementing.

    Args:
        preferred_port: The preferred port to use
        min_port: Minimum port number to try (default: 5000, below this are often privileged)
        max_attempts: Maximum number of ports to try (default: 100)

    Returns:
        An available port number

    Raises:
        RuntimeError: If no available port is found after max_attempts
    """
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


class GenericOpenEnvActor(ForgeActor, Generic[ActT, ObsT]):
    """A generic sandboxed execution environment using any OpenEnv environment.

    This actor provides a universal interface to work with ANY OpenEnv environment
    (CodingEnv, GitEnv, AtariEnv, ChatEnv, etc.) by accepting the environment class
    and action class as parameters.

    It automatically manages the entire container lifecycle including
    image startup, environment connection, and cleanup.

    The actor follows a three-stage workflow:
    1. Environment Initialization: Creates a client from a Docker image.
    2. State Management: Resets the environment for a clean execution state.
    3. Action Execution: Safely executes actions with proper error handling.

    Dependencies:
    - Docker: Must be installed and running on the host.
    - OpenEnv: The OpenEnv library must be in the PYTHONPATH.
    - Docker image: A compatible Docker image for the specific environment.

    Args:
        env_class: The OpenEnv client class (e.g., CodingEnv, GitEnv, AtariEnv).
                  Must be a subclass of HTTPEnvClient[ActT, ObsT].
        action_class: The action class for this environment (e.g., CodeAction, GitAction).
                     Used for type validation and construction helpers.
        docker_image: Docker image to use for the environment (e.g., "coding-env:latest").
        env_vars: Optional dictionary of environment variables to pass to the container.
                 For example: {"PYTHON_ADDITIONAL_IMPORTS": "sys,os,json"} for CodingEnv.
        container_timeout_s: Timeout for container startup in seconds (default: 180.0).
        request_timeout_s: Timeout for individual requests in seconds (default: 120.0).
        port: Preferred port for the container (default: 8000).
        container_memory_gb: Memory limit for the container in GB (default: 4GB).
        enable_zombie_cleanup: Whether to enable zombie process cleanup (default: False).
                              Only relevant for code execution environments.

    Example Usage:

        # CodingEnv example
        from envs.coding_env import CodingEnv, CodeAction

        actor = GenericOpenEnvActor(
            env_class=CodingEnv,
            action_class=CodeAction,
            docker_image="coding-env:latest",
            env_vars={"PYTHON_ADDITIONAL_IMPORTS": "sys,os"},
        )
        await actor.setup()

        action = CodeAction(code="print('hello')")
        result = await actor.execute(action)
        print(result.observation.stdout)

        await actor.teardown()

        # GitEnv example
        from envs.git_env import GitEnv, GitAction

        actor = GenericOpenEnvActor(
            env_class=GitEnv,
            action_class=GitAction,
            docker_image="git-env:latest",
        )
        await actor.setup()

        action = GitAction(action_type="list_repos")
        result = await actor.execute(action)
        print(result.observation.repos)

        await actor.teardown()

        # AtariEnv example
        from envs.atari_env import AtariEnv, AtariAction

        actor = GenericOpenEnvActor(
            env_class=AtariEnv,
            action_class=AtariAction,
            docker_image="atari-env:latest",
        )
        await actor.setup()

        action = AtariAction(action_id=1, game_name="pong")
        result = await actor.execute(action)
        print(f"Reward: {result.reward}, Done: {result.done}")

        await actor.teardown()
    """

    def __init__(
        self,
        env_class: Type[HTTPEnvClient[ActT, ObsT]],
        action_class: Type[ActT],
        docker_image: str,
        env_vars: Optional[Dict[str, str]] = None,
        container_timeout_s: float = 180.0,
        request_timeout_s: float = 120.0,
        port: int = 8000,
        container_memory_gb: int = 4,
        enable_zombie_cleanup: bool = False,
    ):
        self.env_class = env_class
        self.action_class = action_class
        self.docker_image = docker_image
        self.env_vars = env_vars or {}
        self.container_timeout_s = container_timeout_s
        self.request_timeout_s = request_timeout_s
        self.port = port
        self.container_memory_gb = container_memory_gb
        self.enable_zombie_cleanup = enable_zombie_cleanup
        self.client: Optional[HTTPEnvClient[ActT, ObsT]] = None

    @endpoint
    async def setup(self):
        """Initialize the OpenEnv environment and start the container."""
        logging.debug(
            f"Setting up {self.env_class.__name__} actor with image {self.docker_image}"
        )
        try:
            # Find an available port, starting from the preferred port and decrementing
            available_port = find_available_port(self.port)
            if available_port != self.port:
                logger.warning(
                    f"Preferred port {self.port} is in use. Using port {available_port} instead."
                )
            else:
                logger.info(f"Using port {available_port}")

            # Log environment variables if provided
            if self.env_vars:
                logger.debug(
                    f"Passing environment variables to container: {self.env_vars}"
                )

            # Create the environment client using from_docker_image
            # This is universal across all OpenEnv environments
            self.client = self.env_class.from_docker_image(
                self.docker_image,
                timeout_s=self.container_timeout_s,
                request_timeout_s=self.request_timeout_s,
                env_vars=self.env_vars,
                port=available_port,
                memory_gb=self.container_memory_gb,
            )
            logging.debug(f"Successfully initialized {self.env_class.__name__} client.")

            if self.client:
                self.client.reset()
                logging.debug("Initial environment reset complete.")

        except TimeoutError as e:
            logging.error(
                f"Container failed to start within timeout: {e}\n"
                "This may be due to:\n"
                "  1. Docker daemon not running or slow to respond\n"
                "  2. Docker image taking longer than expected to start\n"
                "  3. Container failing to start properly\n"
                "  4. Network/port conflicts\n"
                "Please check Docker logs for more details."
            )
            raise
        except Exception as e:
            logging.error(f"Failed to setup {self.env_class.__name__} client: {e}")
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
    async def execute(self, action: ActT) -> StepResult[ObsT]:
        """Executes an action inside the environment and returns the result.

        Args:
            action: Environment-specific action object (e.g., CodeAction, GitAction, AtariAction).

        Returns:
            StepResult containing the observation, reward, and done flag.
            The observation type depends on the environment (e.g., CodeObservation, GitObservation).

        Raises:
            RuntimeError: If client is not initialized or execution fails after retries.
        """
        logging.debug(f"Executing action: {action}")
        if not self.client:
            raise RuntimeError("Client not initialized. Call setup() first.")

        max_retries = 2
        for attempt in range(max_retries):
            try:
                # Call step() - universal across all OpenEnv environments
                result = self.client.step(action)
                return result

            except Exception as e:
                error_msg = str(e).lower()
                # Check for ACTUAL container-related errors (not HTTP timeouts or code errors)
                # Be specific to avoid false positives that trigger unnecessary container recreation
                is_container_error = any(
                    keyword in error_msg
                    for keyword in [
                        "no such container",
                        "container not found",
                        "container is not running",
                        "container has stopped",
                        "container exited",
                        "exec session",
                        "state improper",
                        "oci runtime error",
                        "docker daemon",
                        "cannot connect to docker",
                        "connection refused",  # Container is dead/not responding
                    ]
                )

                # Exclude HTTP-level errors which don't require container recreation
                is_http_error = any(
                    keyword in error_msg
                    for keyword in [
                        "connection timeout",
                        "read timeout",
                        "http error",
                        "status code",
                    ]
                )

                # Only recreate if it's a real container issue, not just HTTP timeout
                if is_container_error and not (
                    is_http_error and "connection timeout" in error_msg
                ):
                    # ========== CONTAINER ERROR LOGGING ==========
                    # Log to training metrics
                    record_metric(
                        "container/error_count",
                        1,
                        Reduce.SUM,
                    )
                    record_metric(
                        f"container/recreation_attempt_{attempt + 1}",
                        1,
                        Reduce.SUM,
                    )

                    # Prominent console output for debugging
                    print("\n" + "=" * 80)
                    print("🔴 CONTAINER ERROR DETECTED!")
                    print("=" * 80)
                    print(f"Attempt: {attempt + 1}/{max_retries}")
                    print(f"Error Type: {type(e).__name__}")
                    print(f"Error Message: {str(e)}")
                    print(f"Error Message (lowercase): {error_msg}")
                    print(f"\nIs Container Error: {is_container_error}")
                    print(f"Is HTTP Error: {is_http_error}")
                    print("\nFull Traceback:")
                    print("-" * 80)
                    traceback.print_exc()
                    print("-" * 80)
                    print("=" * 80 + "\n")

                    # Also log to logging system
                    logging.error(
                        f"Container error on attempt {attempt + 1}/{max_retries}:\n"
                        f"  Error Type: {type(e).__name__}\n"
                        f"  Error Message: {str(e)}\n"
                        f"  Is Container Error: {is_container_error}\n"
                        f"  Is HTTP Error: {is_http_error}"
                    )

                    if attempt < max_retries - 1:
                        logging.info("Attempting to recreate environment...")
                        try:
                            # Try to recreate the environment with the same env vars and timeouts
                            self.client.close()

                            # Wait for container to fully terminate before creating a new one
                            # This prevents Docker thread exhaustion from rapid container cycling
                            import time

                            logging.info("Waiting for container to fully terminate...")
                            time.sleep(3)  # Give Docker time to clean up

                            # Find an available port for the new container
                            available_port = find_available_port(self.port)
                            if available_port != self.port:
                                logger.warning(
                                    f"Preferred port {self.port} is in use during recreation. "
                                    f"Using port {available_port} instead."
                                )

                            logging.info(
                                f"Recreating container with timeout_s={self.container_timeout_s}, "
                                f"request_timeout_s={self.request_timeout_s}, port={available_port}, "
                                f"memory_gb={self.container_memory_gb}"
                            )
                            self.client = self.env_class.from_docker_image(
                                self.docker_image,
                                timeout_s=self.container_timeout_s,
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
                    # Non-container error, propagate immediately
                    # Log this so user knows why it's not triggering container recreation
                    print("\n" + "-" * 80)
                    print("⚠️  NON-CONTAINER ERROR (will not recreate container)")
                    print("-" * 80)
                    print(f"Error Type: {type(e).__name__}")
                    print(f"Error Message: {str(e)}")
                    print(f"Error Message (lowercase): {error_msg}")
                    print(f"\nIs Container Error: {is_container_error}")
                    print(f"Is HTTP Error: {is_http_error}")
                    print("-" * 80 + "\n")

                    logging.debug(
                        f"Non-container error (propagating immediately):\n"
                        f"  Error Type: {type(e).__name__}\n"
                        f"  Error Message: {str(e)}\n"
                        f"  Is Container Error: {is_container_error}\n"
                        f"  Is HTTP Error: {is_http_error}"
                    )

                    raise

        # Should never reach here, but for type safety
        raise RuntimeError("Execution failed after all retry attempts")

    @endpoint
    async def cleanup_zombie_processes(self) -> int:
        """Kill any zombie python processes in the Docker container that might be consuming memory.

        This is an optional feature primarily useful for code execution environments
        (e.g., CodingEnv, JuliaEnv) where user code might spawn processes that don't terminate.

        For other environment types, this will be skipped if enable_zombie_cleanup is False.

        Returns:
            Number of processes killed
        """
        if not self.enable_zombie_cleanup:
            logging.debug("Zombie cleanup is disabled for this environment")
            return 0

        if not self.client:
            logging.warning("Client not initialized, cannot cleanup zombie processes")
            return 0

        try:
            import subprocess

            # Get container name from the client's container attribute
            container_name = getattr(self.client, "_container_name", None)
            if not container_name:
                logging.warning("Could not determine container name for zombie cleanup")
                return 0

            logging.debug(
                f"Checking for zombie python processes in container {container_name}"
            )

            # Find all python processes in the container (excluding the main server process)
            ps_result = subprocess.run(
                [
                    "docker",
                    "exec",
                    container_name,
                    "sh",
                    "-c",
                    "ps aux | grep python | grep -v 'uvicorn\\|grep' | awk '{print $2}'",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=5,
            )

            if ps_result.returncode == 0 and ps_result.stdout.strip():
                pids = [pid for pid in ps_result.stdout.strip().split("\n") if pid]
                if len(pids) > 1:  # More than just the main process
                    pid_count = len(pids) - 1  # Exclude main server process
                    logging.warning(
                        f"Found {pid_count} potential zombie python processes in container, killing them"
                    )

                    # Kill zombie processes
                    for pid in pids[1:]:  # Skip first PID (main server)
                        subprocess.run(
                            [
                                "docker",
                                "exec",
                                container_name,
                                "kill",
                                "-9",
                                pid,
                            ],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            timeout=2,
                        )

                    logging.info(f"Successfully killed {pid_count} zombie processes")
                    record_metric(
                        "container/zombie_processes_killed", pid_count, Reduce.SUM
                    )
                    return pid_count
                else:
                    logging.debug(
                        "No zombie processes found (only main server process)"
                    )
                    return 0
            else:
                logging.debug("No zombie processes found")
                return 0

        except Exception as e:
            logging.error(f"Error during zombie process cleanup: {e}")
            return 0

    @endpoint
    async def get_state(self) -> Any:
        """Get the current environment state.

        Returns:
            Environment-specific state object (e.g., CodeState, GitState, AtariState).
        """
        if not self.client:
            raise RuntimeError("Client not initialized. Call setup() first.")
        return self.client.state()

    @endpoint
    async def teardown(self):
        """Cleans up the environment and stops the container."""
        if self.client:
            logging.debug(
                f"Closing {self.env_class.__name__} client and stopping container."
            )

            # CRITICAL: Clean up any zombie processes before closing if enabled
            if self.enable_zombie_cleanup:
                try:
                    killed_count = await self.cleanup_zombie_processes()
                    if killed_count > 0:
                        logging.info(
                            f"Cleaned up {killed_count} zombie processes before teardown"
                        )
                except Exception as e:
                    logging.error(
                        f"Error cleaning zombie processes during teardown: {e}"
                    )

            # Close the client which stops and removes the container
            self.client.close()
            self.client = None
            logging.debug("Cleanup complete.")

    def create_action(self, **kwargs) -> ActT:
        """Helper method to create an action for this environment.

        Args:
            **kwargs: Arguments to pass to the action class constructor.

        Returns:
            An instance of the environment's action class.

        Example:
            actor = GenericOpenEnvActor(env_class=CodingEnv, action_class=CodeAction, ...)
            action = actor.create_action(code="print('hello')")
        """
        return self.action_class(**kwargs)
