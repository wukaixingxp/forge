# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Unit tests for forge.actors.coder.SandboxedPythonCoder.
"""
import uuid
from contextlib import asynccontextmanager
from unittest.mock import Mock, patch

import pytest
from forge.actors.coder import SandboxedPythonCoder

from monarch.actor import this_proc


@asynccontextmanager
async def create_mock_coder(
    execute_stdout="hello world\n",
    execute_returncode=0,
    execute_stderr="",
    pull_fails=False,
    create_fails=False,
    start_fails=False,
):
    """Context manager that creates a mocked SandboxedPythonCoder."""
    unique_id = str(uuid.uuid4())[:8]
    container_name = f"test_sandbox_{unique_id}"
    container_image = "python:3.10"

    coder = None
    try:
        with patch("subprocess.run") as mock_run:

            def mock_subprocess_run(*args, **kwargs):
                cmd = args[0]
                # Check if image exists locally
                if "image" in cmd and "exists" in cmd:
                    result = Mock()
                    result.returncode = 1  # Image doesn't exist, will trigger pull
                    result.stderr = ""
                    return result
                # Pull image
                elif "pull" in cmd:
                    result = Mock()
                    if pull_fails:
                        result.returncode = 1
                        result.stderr = "Failed to pull image: network error"
                    else:
                        result.returncode = 0
                        result.stderr = ""
                        result.stdout = f"Pulling {container_image}..."
                    return result
                # Remove container
                elif "rm" in cmd:
                    result = Mock()
                    result.returncode = 0
                    result.stderr = ""
                    return result
                # Create container
                elif "create" in cmd:
                    result = Mock()
                    if create_fails:
                        result.returncode = 1
                        result.stderr = "Failed to create container: no space"
                    else:
                        result.returncode = 0
                        result.stderr = ""
                        result.stdout = container_name
                    return result
                # Start container
                elif "start" in cmd and "exec" not in cmd:
                    result = Mock()
                    if start_fails:
                        result.returncode = 1
                        result.stderr = "Failed to start container"
                    else:
                        result.returncode = 0
                        result.stderr = ""
                        result.stdout = container_name
                    return result
                # Copy file to container
                elif "cp" in cmd:
                    result = Mock()
                    result.returncode = 0
                    result.stderr = ""
                    result.stdout = ""
                    return result
                # Execute command in container
                elif "exec" in cmd:
                    result = Mock()
                    result.returncode = execute_returncode
                    result.stdout = execute_stdout
                    result.stderr = execute_stderr
                    return result
                else:
                    raise ValueError(f"Unexpected subprocess call: {cmd}")

            mock_run.side_effect = mock_subprocess_run

            coder = this_proc().spawn(
                f"coder_{uuid.uuid1()}",
                SandboxedPythonCoder,
                container_image,
                container_name,
            )

            yield coder, mock_run

    finally:
        if coder:
            await SandboxedPythonCoder.shutdown(coder)


@pytest.mark.timeout(10)
@pytest.mark.asyncio
async def test_coder_success():
    """Test successful execution."""
    async with create_mock_coder(execute_stdout="Hello World\n") as (coder, mock_run):
        await coder.setup.call_one()
        result, _ = await coder.execute.call_one(code="print('Hello World')")
        assert result == "Hello World\n"
        
        # Verify proper sequence of podman commands
        calls = [str(call[0][0]) for call in mock_run.call_args_list]
        # Should have image exists check, pull, rm, create, start, cp, exec
        assert any("image" in call and "exists" in call for call in calls)
        assert any("pull" in call for call in calls)
        assert any("create" in call for call in calls)
        assert any("start" in call for call in calls)
        assert any("cp" in call for call in calls)
        assert any("exec" in call for call in calls)


@pytest.mark.timeout(10)
@pytest.mark.asyncio
async def test_coder_execution_failure():
    """Test execution failure."""
    async with create_mock_coder(
        execute_returncode=1, execute_stderr="SyntaxError: invalid syntax"
    ) as (coder, _):
        await coder.setup.call_one()
        output, err = await coder.execute.call_one(code="invalid syntax")
        assert "SyntaxError" in err


@pytest.mark.timeout(10)
@pytest.mark.asyncio
async def test_coder_pull_failure():
    """Test image pull failure."""
    async with create_mock_coder(pull_fails=True) as (coder, _):
        with pytest.raises(Exception, match="Failed to pull image with podman"):
            await coder.setup.call_one()


@pytest.mark.timeout(10)
@pytest.mark.asyncio
async def test_coder_create_failure():
    """Test container creation failure."""
    async with create_mock_coder(create_fails=True) as (coder, _):
        with pytest.raises(Exception, match="Failed to recreate container"):
            await coder.setup.call_one()


@pytest.mark.timeout(10)
@pytest.mark.asyncio
async def test_coder_start_failure():
    """Test container start failure."""
    async with create_mock_coder(start_fails=True) as (coder, _):
        with pytest.raises(Exception, match="Failed to start container:"):
            await coder.setup.call_one()


@pytest.mark.timeout(10)
@pytest.mark.asyncio
async def test_coder_recreate():
    """Test container recreation."""
    async with create_mock_coder() as (coder, mock_run):
        await coder.setup.call_one()
        
        # Reset mock to count new calls
        mock_run.reset_mock()
        
        # Recreate the container
        await coder.recreate.call_one()
        
        # Should remove, create, and start a new container
        calls = [str(call[0][0]) for call in mock_run.call_args_list]
        assert any("rm" in call for call in calls)
        assert any("create" in call for call in calls)
        assert any("start" in call for call in calls)


@pytest.mark.timeout(10)
@pytest.mark.asyncio
async def test_coder_execute_without_setup():
    """Test execution without setup raises error."""
    async with create_mock_coder() as (coder, _):
        # Try to execute without setup
        with pytest.raises(Exception, match="Container not initialized\\. Call recreate\\(\\) first\\."):
            await coder.execute.call_one(code="print('Hello')")
