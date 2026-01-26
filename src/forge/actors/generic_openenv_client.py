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
from typing import Any, Dict, Optional, TYPE_CHECKING

from monarch.actor import endpoint

if TYPE_CHECKING:
    from openenv import GenericEnvClient, GenericAction
    from openenv.core.client_types import StepResult

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

    This actor can manage multiple WebSocket connections to one or more containers,
    allowing concurrent requests to fully utilize the Julia worker pool.

    Benefits:
    - No need to import environment-specific packages
    - Works with any OpenEnv-compatible Docker image
    - Connection pooling for high concurrency
    - Load balancing across multiple containers

    Usage:
        >>> actor = GenericOpenEnvClientActor(
        ...     docker_image="julia-env:latest",
        ...     num_connections=16,  # 16 concurrent WebSocket connections
        ...     num_containers=2,     # Spread across 2 containers
        ... )
        >>> await actor.setup()
        >>> action = {"core_code": "...", "test_code": "..."}
        >>> result = await actor.execute(action)  # Uses available connection from pool
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
        num_connections: int = 1,
        num_containers: int = 1,
    ):
        """Initialize the generic OpenEnv actor with connection pooling.

        Args:
            docker_image: Docker image name (e.g., "julia-env:latest")
            env_vars: Environment variables to pass to containers
            container_timeout_s: Timeout for container startup in seconds
            request_timeout_s: Timeout for individual requests in seconds
            port: Starting port for containers
            container_memory_gb: Memory limit per container in GB
            enable_zombie_cleanup: Whether to enable zombie process cleanup
            num_connections: Total number of WebSocket connections to create
            num_containers: Number of Docker containers to distribute connections across
        """
        self.docker_image = docker_image
        self.env_vars = env_vars or {}
        self.container_timeout_s = container_timeout_s
        self.request_timeout_s = request_timeout_s
        self.port = port
        self.container_memory_gb = container_memory_gb
        self.enable_zombie_cleanup = enable_zombie_cleanup
        self.num_connections = num_connections
        self.num_containers = num_containers

        # Connection pool
        self.clients: list = []  # List of GenericEnvClient instances
        self.client_available: list = []  # Availability flags
        self.client_lock = None  # asyncio.Lock, initialized in setup
        self.client_condition = None  # asyncio.Condition for waiting on available clients

        # For backward compatibility
        self.client = None
        self.actual_port = None

    @endpoint
    async def setup(self):
        """Initialize containers and create connection pool."""
        from openenv import GenericEnvClient
        from openenv.core.containers.runtime import LocalDockerProvider
        import os
        import asyncio

        self.client_lock = asyncio.Lock()
        self.client_condition = asyncio.Condition(self.client_lock)

        logger.info(
            f"Setting up connection pool: {self.num_connections} connections "
            f"across {self.num_containers} containers"
        )

        # Setup persistent logging with volume mount
        logs_dir = os.path.expanduser("~/julia_container_logs")
        os.makedirs(logs_dir, exist_ok=True)

        # Step 1: Create Docker containers
        self.providers = []
        self.container_urls = []

        for i in range(self.num_containers):
            try:
                # Find available port for this container
                container_port = find_available_port(self.port - i)

                logger.info(f"Creating container {i+1}/{self.num_containers} on port {container_port}")

                # Setup environment variables for this container
                container_env_vars = self.env_vars.copy()
                container_env_vars["PORT"] = str(container_port)

                # Setup logging for this container
                log_filename = f"julia_env_port_{container_port}.log"
                container_log_path = f"/tmp/julia_logs/{log_filename}"
                container_env_vars["JULIA_LOG_FILE"] = container_log_path
                container_env_vars["JULIA_LOG_LEVEL"] = "DEBUG"

                volumes = {logs_dir: "/tmp/julia_logs"}

                # Create container provider
                provider = LocalDockerProvider()

                # Start container and get base URL
                base_url = provider.start_container(
                    self.docker_image,
                    port=container_port,
                    env_vars=container_env_vars,
                    volumes=volumes,
                    memory_gb=self.container_memory_gb,
                )

                # Wait for container to be ready
                provider.wait_for_ready(base_url, timeout_s=self.container_timeout_s)

                self.providers.append(provider)
                self.container_urls.append(base_url)

                logger.info(
                    f"Container {i+1}/{self.num_containers} ready at {base_url}, "
                    f"logs: {logs_dir}/{log_filename}"
                )

            except Exception as e:
                logger.error(f"Failed to create container {i+1}: {e}")
                # Cleanup already created containers
                for provider in self.providers:
                    try:
                        provider.stop_container()
                    except Exception:
                        pass
                raise

        # Step 2: Create WebSocket connections distributed across containers
        logger.info(f"Creating {self.num_connections} WebSocket connections...")

        for i in range(self.num_connections):
            try:
                # Round-robin: which container should this connection use?
                container_idx = i % self.num_containers
                base_url = self.container_urls[container_idx]

                logger.debug(
                    f"Creating connection {i+1}/{self.num_connections} → "
                    f"container {container_idx} ({base_url})"
                )

                # Create client connecting to existing container
                client = GenericEnvClient(
                    base_url=base_url,
                    connect_timeout_s=10.0,
                    message_timeout_s=self.request_timeout_s,
                )
                client.connect()

                # Reset the environment for this connection
                client.reset()

                self.clients.append(client)
                self.client_available.append(True)

                logger.debug(f"Connection {i+1}/{self.num_connections} established")

            except Exception as e:
                logger.error(f"Failed to create connection {i+1}: {e}")
                # Cleanup
                for client in self.clients:
                    try:
                        client.close()
                    except Exception:
                        pass
                for provider in self.providers:
                    try:
                        provider.stop_container()
                    except Exception:
                        pass
                raise

        logger.info(
            f"Connection pool ready: {len(self.clients)} connections across "
            f"{len(self.container_urls)} containers"
        )

        # Set self.client to the first pooled client for backward compatibility
        # This ensures execute() works with the pool
        if self.clients:
            self.client = self.clients[0]
            self.actual_port = self.port  # Use the base port for identification
            logger.info(f"Primary client set from connection pool (port {self.port})")
        else:
            raise RuntimeError("No clients were created in the connection pool")

    async def _acquire_client(self, timeout: float = 30.0) -> tuple:
        """Acquire an available client from the connection pool.

        Args:
            timeout: Maximum time to wait for an available client in seconds.

        Returns:
            Tuple of (client_index, client)

        Raises:
            TimeoutError: If no client becomes available within timeout.
        """
        import asyncio

        start_time = asyncio.get_event_loop().time()

        async with self.client_condition:
            while True:
                # Find available client
                for i, available in enumerate(self.client_available):
                    if available:
                        self.client_available[i] = False
                        logger.debug(f"Acquired client {i} from pool")
                        return i, self.clients[i]

                # Check timeout
                elapsed = asyncio.get_event_loop().time() - start_time
                remaining = timeout - elapsed
                if remaining <= 0:
                    raise TimeoutError(
                        f"No client available in pool after {timeout}s. "
                        f"All {len(self.clients)} clients are busy."
                    )

                # Wait for a client to become available
                logger.debug(f"All clients busy, waiting for availability (timeout: {remaining:.1f}s)")
                try:
                    await asyncio.wait_for(
                        self.client_condition.wait(),
                        timeout=remaining
                    )
                except asyncio.TimeoutError:
                    raise TimeoutError(
                        f"No client available in pool after {timeout}s. "
                        f"All {len(self.clients)} clients are busy."
                    )

    async def _release_client(self, client_idx: int):
        """Release a client back to the connection pool.

        Args:
            client_idx: Index of the client to release.
        """
        async with self.client_condition:
            self.client_available[client_idx] = True
            logger.debug(f"Released client {client_idx} back to pool")
            self.client_condition.notify()  # Wake up one waiting acquirer

    async def _reconnect_client(self, client_idx: int) -> "GenericEnvClient":
        """Reconnect a failed client to its container.

        Args:
            client_idx: Index of the client to reconnect.

        Returns:
            The new client instance.
        """
        from openenv import GenericEnvClient

        # Determine which container this client was connected to
        container_idx = client_idx % self.num_containers
        base_url = self.container_urls[container_idx]

        logger.info(f"Reconnecting client {client_idx} to container {container_idx} ({base_url})")

        # Close old client
        try:
            old_client = self.clients[client_idx]
            if old_client:
                old_client.close()
        except Exception as e:
            logger.debug(f"Error closing old client: {e}")

        # Create new connection to same container
        import time
        time.sleep(1)  # Brief pause before reconnect

        new_client = GenericEnvClient(
            base_url=base_url,
            connect_timeout_s=10.0,
            message_timeout_s=self.request_timeout_s,
        )
        new_client.connect()
        new_client.reset()

        # Update pool
        self.clients[client_idx] = new_client
        logger.info(f"Client {client_idx} reconnected successfully")

        record_metric("pool/reconnect_success", 1, Reduce.SUM)
        return new_client

    @endpoint
    async def recreate(self):
        """Resets the environment to a clean state."""
        if not self.client:
            raise RuntimeError("Client not initialized. Call setup() first.")
        logging.debug("Recreating environment state (resetting).")
        self.client.reset()
        logging.debug("Environment reset.")

    @endpoint
    async def execute(self, action: Dict[str, Any]) -> "StepResult[Dict[str, Any]]":
        """Executes an action using an available connection from the pool.

        This method acquires a client from the connection pool, executes the action,
        and releases the client back to the pool. Multiple concurrent calls will
        be distributed across available connections.

        Args:
            action: Dictionary action (or GenericAction). For Julia environments,
                   this should contain {"core_code": "...", "test_code": "..."}

        Returns:
            StepResult containing:
                - observation: Dict with keys like exit_code, stdout, stderr,
                              tests_passed, tests_failed, code_compiles, etc.
        """
        logging.debug(f"Executing action (pool size: {len(self.clients)})")

        if not self.clients:
            raise RuntimeError("Connection pool not initialized. Call setup() first.")

        # Acquire a client from the pool
        client_idx, client = await self._acquire_client(timeout=self.request_timeout_s)

        try:
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    result = client.step(action)
                    record_metric("pool/execute_success", 1, Reduce.SUM)
                    return result

                except Exception as e:
                    error_msg = str(e).lower()

                    # Check for WebSocket connection errors (indicates container crash)
                    is_websocket_error = any(
                        keyword in error_msg
                        for keyword in [
                            "connectionclosederror",
                            "keepalive ping timeout",
                            "websocket",
                            "connection closed",
                            "connection reset",
                            "broken pipe",
                        ]
                    )

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

                    if is_websocket_error or is_container_error:
                        error_type = "WebSocket" if is_websocket_error else "Container"
                        record_metric(f"pool/{error_type.lower()}_error_count", 1, Reduce.SUM)
                        logging.error(
                            f"{error_type} error on client {client_idx}, "
                            f"attempt {attempt + 1}/{max_retries}: {e}"
                        )

                        if attempt < max_retries - 1:
                            logging.info(
                                f"Reconnecting client {client_idx} "
                                f"(attempt {attempt + 2}/{max_retries})..."
                            )
                            try:
                                client = await self._reconnect_client(client_idx)
                                continue
                            except Exception as reconnect_error:
                                logging.error(
                                    f"Failed to reconnect client {client_idx}: {reconnect_error}"
                                )
                                record_metric("pool/reconnect_failure", 1, Reduce.SUM)
                                if attempt == max_retries - 1:
                                    raise RuntimeError(
                                        f"Client {client_idx} connection failed after "
                                        f"{max_retries} attempts. Last error: {e}"
                                    ) from e
                        else:
                            raise RuntimeError(
                                f"Client {client_idx} connection failed after "
                                f"{max_retries} attempts: {e}"
                            ) from e
                    else:
                        # Non-connection error, don't retry
                        raise

            raise RuntimeError("Execution failed after all retry attempts")

        finally:
            # Always release client back to pool
            await self._release_client(client_idx)

    @endpoint
    async def health_check(self) -> Dict[str, Any]:
        """Check if the environment is healthy and responsive.

        Returns:
            Dict with:
                - healthy: True if environment is responsive
                - error: Error message if unhealthy
                - pool_status: Status of each client in the pool
        """
        if not self.clients:
            return {"healthy": False, "error": "Connection pool not initialized", "pool_status": []}

        pool_status = []
        healthy_count = 0
        for i, client in enumerate(self.clients):
            try:
                client.state()
                pool_status.append({"index": i, "healthy": True, "available": self.client_available[i]})
                healthy_count += 1
            except Exception as e:
                pool_status.append({"index": i, "healthy": False, "error": str(e), "available": self.client_available[i]})

        return {
            "healthy": healthy_count > 0,
            "error": None if healthy_count > 0 else "All clients unhealthy",
            "healthy_count": healthy_count,
            "total_clients": len(self.clients),
            "available_clients": sum(1 for a in self.client_available if a),
            "pool_status": pool_status,
        }

    @endpoint
    async def get_pool_status(self) -> Dict[str, Any]:
        """Get the current connection pool status.

        Returns:
            Dict with pool statistics.
        """
        return {
            "num_containers": len(self.container_urls),
            "num_connections": len(self.clients),
            "available_connections": sum(1 for a in self.client_available if a),
            "busy_connections": sum(1 for a in self.client_available if not a),
            "container_urls": self.container_urls,
        }

    @endpoint
    async def get_state(self) -> Dict[str, Any]:
        """Get the current environment state as a dictionary."""
        if not self.clients:
            raise RuntimeError("Connection pool not initialized. Call setup() first.")
        # Use first available client to get state
        return self.clients[0].state()

    @endpoint
    async def teardown(self):
        """Cleans up all connections and stops all containers."""
        logging.debug("Tearing down connection pool...")

        # Close all clients
        for i, client in enumerate(self.clients):
            try:
                client.close()
                logging.debug(f"Closed client {i}")
            except Exception as e:
                logging.warning(f"Error closing client {i}: {e}")

        self.clients = []
        self.client_available = []
        self.client = None

        # Stop all containers
        for i, provider in enumerate(self.providers):
            try:
                provider.stop_container()
                logging.debug(f"Stopped container {i}")
            except Exception as e:
                logging.warning(f"Error stopping container {i}: {e}")

        self.providers = []
        self.container_urls = []

        logging.debug("Connection pool teardown complete.")

    @endpoint
    async def restart_container(self) -> Dict[str, Any]:
        """Restart ALL containers and reconnect the entire connection pool.

        This is used by the circuit breaker when containers become unhealthy.
        It properly tears down all existing containers and clients, then
        recreates the entire pool from scratch.

        Returns:
            Dict with 'success' boolean and 'error' message if failed
        """
        import os
        import asyncio
        import subprocess
        from openenv import GenericEnvClient
        from openenv.core.containers.runtime import LocalDockerProvider

        logger.warning(
            f"Restarting ALL containers and connection pool "
            f"(had {len(getattr(self, 'providers', []))} containers, "
            f"{len(getattr(self, 'clients', []))} connections)"
        )

        try:
            # Step 1: Close all pooled clients
            for i, client in enumerate(getattr(self, 'clients', [])):
                try:
                    client.close()
                    logger.debug(f"Closed client {i}")
                except Exception as e:
                    logger.debug(f"Error closing client {i}: {e}")

            self.clients = []
            self.client_available = []
            self.client = None

            # Step 2: Stop all existing containers (with timeout to avoid hanging)
            for i, provider in enumerate(getattr(self, 'providers', [])):
                try:
                    # Use subprocess with timeout to avoid hanging on docker stop
                    if hasattr(provider, 'container_id') and provider.container_id:
                        try:
                            subprocess.run(
                                ['docker', 'stop', provider.container_id],
                                timeout=10,
                                capture_output=True
                            )
                        except subprocess.TimeoutExpired:
                            # Force kill if stop times out
                            subprocess.run(
                                ['docker', 'kill', provider.container_id],
                                timeout=5,
                                capture_output=True
                            )
                    else:
                        provider.stop_container()
                    logger.debug(f"Stopped container {i}")
                except Exception as e:
                    logger.warning(f"Error stopping container {i}: {e}")

            self.providers = []
            self.container_urls = []

            # Wait for cleanup
            await asyncio.sleep(2)

            # Step 3: Setup persistent logging
            logs_dir = os.path.expanduser("~/julia_container_logs")
            os.makedirs(logs_dir, exist_ok=True)

            # Step 4: Create new containers
            new_providers = []
            new_container_urls = []

            for i in range(self.num_containers):
                try:
                    # Find available port
                    container_port = find_available_port(self.port - i)

                    logger.info(f"Restart: Creating container {i+1}/{self.num_containers} on port {container_port}")

                    # Setup environment variables
                    container_env_vars = self.env_vars.copy()
                    container_env_vars["PORT"] = str(container_port)

                    # Setup logging
                    log_filename = f"julia_env_port_{container_port}.log"
                    container_log_path = f"/tmp/julia_logs/{log_filename}"
                    container_env_vars["JULIA_LOG_FILE"] = container_log_path
                    container_env_vars["JULIA_LOG_LEVEL"] = "DEBUG"

                    volumes = {logs_dir: "/tmp/julia_logs"}

                    # Create container provider
                    provider = LocalDockerProvider()
                    base_url = provider.start_container(
                        self.docker_image,
                        port=container_port,
                        env_vars=container_env_vars,
                        volumes=volumes,
                        memory_gb=self.container_memory_gb,
                    )

                    # Wait for container to be ready
                    provider.wait_for_ready(base_url, timeout_s=self.container_timeout_s)

                    new_providers.append(provider)
                    new_container_urls.append(base_url)

                    logger.info(f"Restart: Container {i+1}/{self.num_containers} ready at {base_url}")

                except Exception as e:
                    logger.error(f"Restart: Failed to create container {i+1}: {e}")
                    # Cleanup partially created containers
                    for p in new_providers:
                        try:
                            p.stop_container()
                        except Exception:
                            pass
                    return {"success": False, "error": f"Failed to create container {i+1}: {e}"}

            self.providers = new_providers
            self.container_urls = new_container_urls

            # Step 5: Create new WebSocket connections
            new_clients = []
            new_client_available = []

            for i in range(self.num_connections):
                try:
                    container_idx = i % self.num_containers
                    base_url = self.container_urls[container_idx]

                    logger.debug(f"Restart: Creating connection {i+1}/{self.num_connections} → container {container_idx}")

                    client = GenericEnvClient(
                        base_url=base_url,
                        connect_timeout_s=10.0,
                        message_timeout_s=self.request_timeout_s,
                    )
                    client.connect()
                    client.reset()

                    new_clients.append(client)
                    new_client_available.append(True)

                except Exception as e:
                    logger.error(f"Restart: Failed to create connection {i+1}: {e}")
                    # Cleanup
                    for c in new_clients:
                        try:
                            c.close()
                        except Exception:
                            pass
                    for p in self.providers:
                        try:
                            p.stop_container()
                        except Exception:
                            pass
                    return {"success": False, "error": f"Failed to create connection {i+1}: {e}"}

            self.clients = new_clients
            self.client_available = new_client_available

            # Set primary client for backward compatibility
            if self.clients:
                self.client = self.clients[0]
                self.actual_port = self.port

            logger.info(
                f"Restart: Pool restored with {len(self.clients)} connections "
                f"across {len(self.container_urls)} containers"
            )
            return {
                "success": True,
                "error": None,
                "num_containers": len(self.container_urls),
                "num_connections": len(self.clients),
            }

        except Exception as e:
            logger.error(f"Failed to restart containers: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}

    def create_action(self, **kwargs) -> "GenericAction":
        """Helper method to create a GenericAction.

        Args:
            **kwargs: Arguments for the action (e.g., core_code, test_code)

        Returns:
            GenericAction instance (dictionary wrapper)

        Example:
            action = actor.create_action(core_code="println(1)", test_code="@test true")
        """
        from openenv import GenericAction

        return GenericAction(**kwargs)
