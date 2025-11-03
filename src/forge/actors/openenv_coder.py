# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import socket
import sys
import traceback

from envs.coding_env import CodeAction, CodingEnv

from forge.controller import ForgeActor
from forge.observability.metrics import record_metric, Reduce
from monarch.actor import endpoint

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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
        additional_imports: List of additional Python modules to authorize for import.
                          These will be passed to the container via the PYTHON_ADDITIONAL_IMPORTS
                          environment variable. Default: ["sys", "os", "functools", "typing"]
        container_memory_gb: Memory limit for the container in GB (default: 4GB).
                           Set this based on your system resources to prevent OOM kills.
    """

    def __init__(
        self,
        docker_image: str = "coding-env:latest",
        additional_imports: list[str] | None = None,
        container_timeout_s: float = 180.0,
        request_timeout_s: float = 120.0,
        port: int = 8432,
        container_memory_gb: int = 4,
    ):
        self.docker_image = docker_image
        if additional_imports is None:
            # Default imports that match common_imports in reward evaluation
            additional_imports = ["sys", "os", "functools", "typing"]
        self.additional_imports = additional_imports
        self.container_timeout_s = container_timeout_s
        self.request_timeout_s = request_timeout_s
        self.port = port
        self.container_memory_gb = container_memory_gb
        self.client: CodingEnv | None = None

    @endpoint
    async def setup(self):
        logging.debug(f"Setting up OpenEnv actor with image {self.docker_image}")
        try:
            # Find an available port, starting from the preferred port and decrementing
            available_port = find_available_port(self.port)
            if available_port != self.port:
                logger.warning(
                    f"Preferred port {self.port} is in use. Using port {available_port} instead."
                )
            else:
                logger.info(f"Using port {available_port}")

            # Prepare environment variables for the container
            env_vars = {}
            if self.additional_imports:
                # Convert list to comma-separated string for environment variable
                imports_str = ",".join(self.additional_imports)
                env_vars["PYTHON_ADDITIONAL_IMPORTS"] = imports_str
                logging.debug(f"Passing additional imports to container: {imports_str}")

            # Use a longer timeout to allow container to fully start
            # Some Docker images can take longer to initialize
            # Also set a longer request timeout for long-running code execution
            self.client = CodingEnv.from_docker_image(
                self.docker_image,
                timeout_s=self.container_timeout_s,
                request_timeout_s=self.request_timeout_s,
                env_vars=env_vars,
                port=available_port,
                memory_gb=self.container_memory_gb,  # Pass memory limit to container
            )
            logging.debug("Successfully initialized OpenEnv client.")
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
            logging.error(f"Failed to setup OpenEnv client: {e}")
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
    async def execute(self, code: str) -> tuple[str, str]:
        """Executes Python code inside the environment and returns the output.

        Args:
            code: Python source code string to execute.

        Returns:
            The captured stdout and stderr from the execution, as a
            (stdout, stderr) tuple of strings.
        """
        logging.debug(f"Executing {code}")
        if not self.client:
            raise RuntimeError("Client not initialized. Call setup() first.")

        max_retries = 2
        for attempt in range(max_retries):
            try:
                result = self.client.step(CodeAction(code=code))

                output = result.observation.stdout
                error = result.observation.stderr

                return output, error

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
                            # This prevents Podman thread exhaustion from rapid container cycling
                            import time

                            logging.info("Waiting for container to fully terminate...")
                            time.sleep(3)  # Give Podman time to clean up

                            # Find an available port for the new container
                            available_port = find_available_port(self.port)
                            if available_port != self.port:
                                logger.warning(
                                    f"Preferred port {self.port} is in use during recreation. "
                                    f"Using port {available_port} instead."
                                )

                            env_vars = {}
                            if self.additional_imports:
                                imports_str = ",".join(self.additional_imports)
                                env_vars["PYTHON_ADDITIONAL_IMPORTS"] = imports_str
                            logging.info(
                                f"Recreating container with timeout_s={self.container_timeout_s}, "
                                f"request_timeout_s={self.request_timeout_s}, port={available_port}, "
                                f"memory_gb={self.container_memory_gb}"
                            )
                            self.client = CodingEnv.from_docker_image(
                                self.docker_image,
                                timeout_s=self.container_timeout_s,
                                request_timeout_s=self.request_timeout_s,
                                env_vars=env_vars,
                                port=available_port,
                                memory_gb=self.container_memory_gb,  # Pass memory limit to recreated container
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
    async def teardown(self):
        """Cleans up the environment and stops the container."""
        if self.client:
            logging.debug("Closing OpenEnv client and stopping container.")
            self.client.close()
            self.client = None
            logging.debug("Cleanup complete.")
