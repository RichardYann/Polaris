from app.services.managed_commands import (
    OperationContext,
    RepairScope,
    failure_from_exception,
)
from app.services.managed_ssh import ManagedCommandLaunchError, SSHManagedCommands
from tests.fake_ssh import FakeSSHServer, FakeSSHSession


def _manager(server: FakeSSHServer) -> SSHManagedCommands:
    session = FakeSSHSession(server)
    return SSHManagedCommands(
        session=session,
        run=session.run,
        shell_workdir="~/polaris_runs/test",
        sftp_workdir="polaris_runs/test",
    )


def _context() -> OperationContext:
    return OperationContext(
        phase="application.run",
        operation="experiment-run",
        display_command="bash run.sh",
        target="host",
        repair_scope=RepairScope.APPLICATION_FILES,
    )


async def test_concurrent_logical_start_attaches_active_attempt():
    server = FakeSSHServer(run_exit=None)
    manager = _manager(server)

    first = await manager.start(_context(), "bash run.sh")
    second = await manager.start(_context(), "bash run.sh")

    assert second.attempt_id == first.attempt_id
    assert second.process_id == first.process_id
    assert sum("nohup setsid bash" in command for command in server.commands) == 1
    pointer_index = next(
        i for i, command in enumerate(server.commands) if "/current.tmp" in command
    )
    launch_index = next(
        i for i, command in enumerate(server.commands) if "nohup setsid bash" in command
    )
    assert pointer_index < launch_index
    assert "&& nohup" not in server.commands[launch_index]


async def test_completed_attempt_allows_a_new_attempt():
    server = FakeSSHServer(run_exit=0)
    manager = _manager(server)

    first = await manager.start(_context(), "bash run.sh")
    second = await manager.start(_context(), "bash run.sh")

    assert second.attempt_id != first.attempt_id
    assert sum("nohup setsid bash" in command for command in server.commands) == 2


async def test_stop_only_targets_current_attempt_and_verifies_exit():
    server = FakeSSHServer(run_exit=None)
    manager = _manager(server)
    handle = await manager.start(_context(), "bash run.sh")

    assert await manager.stop(handle) is True
    assert handle.process_group_id in server.killed
    snapshot = await manager.snapshot(handle)
    assert snapshot.process_alive is False
    assert snapshot.exit_status == -15


async def test_launch_timeout_recovers_the_durable_attempt():
    server = FakeSSHServer(run_exit=None)
    session = FakeSSHSession(server)

    class LaunchTimeout(TimeoutError):
        def __init__(self, stdout: str) -> None:
            self.stdout = stdout
            self.stderr = ""
            self.exit_status = 0
            super().__init__("launch channel timed out after returning the PID")

    async def run(command: str, timeout: float | None = None):
        result = await session.run(command, timeout)
        if "nohup setsid bash" in command:
            raise LaunchTimeout(result.stdout)
        return result

    manager = SSHManagedCommands(
        session=session,
        run=run,
        shell_workdir="~/polaris_runs/test",
        sftp_workdir="polaris_runs/test",
    )
    handle = await manager.start(_context(), "bash run.sh")

    assert handle.process_id == server.pid
    assert await manager.current_attempt_id(handle.operation_id) == handle.attempt_id
    assert sum("nohup setsid bash" in command for command in server.commands) == 1


def test_launch_error_preserves_operation_context():
    context = _context()
    error = ManagedCommandLaunchError(context, "bash run.sh", TimeoutError())
    assert error.operation_context is context
    assert error.command == "bash run.sh"
    report = failure_from_exception(error)
    assert report.phase == "application.run"
    assert report.operation == "experiment-run"
    assert report.repair_scope == RepairScope.APPLICATION_FILES
