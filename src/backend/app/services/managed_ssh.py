"""Generic durable command execution over an existing SSH session.

The backend persists one uniform process envelope for every command.  It does
not inspect command names and does not need per-command lifecycle adapters.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol

from app.services.managed_commands import CommandSnapshot, OperationContext, redact_text

_OPERATION_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


class SSHResultLike(Protocol):
    exit_status: int
    stdout: str
    stderr: str


class SSHSessionLike(Protocol):
    async def write_file(self, path: str, content: str) -> None: ...


RunCommand = Callable[[str, float | None], Awaitable[SSHResultLike]]


@dataclass(slots=True, frozen=True)
class ManagedCommandHandle:
    operation_id: str
    attempt_id: str
    context: OperationContext
    process_id: int
    process_group_id: int


@dataclass(slots=True, frozen=True)
class OutputChunk:
    stream: str
    text: str
    offset: int


class ManagedCommandLaunchError(RuntimeError):
    """A managed operation failed before a durable handle could be recovered."""

    def __init__(
        self,
        context: OperationContext,
        command: str,
        cause: BaseException,
    ) -> None:
        self.operation_context = context
        self.command = redact_text(context.display_command or command)
        self.stdout = redact_text(getattr(cause, "stdout", ""), tail=True)
        self.stderr = redact_text(getattr(cause, "stderr", ""), tail=True)
        self.exit_status = getattr(cause, "exit_status", None)
        self.original_exception = cause
        detail = str(cause).strip() or self.stderr or self.stdout or type(cause).__name__
        super().__init__(f"managed command launch failed: {detail}")


class SSHManagedCommands:
    """Persist, inspect, stream, and stop process-group-bound commands."""

    def __init__(
        self,
        *,
        session: SSHSessionLike,
        run: RunCommand,
        shell_workdir: str,
        sftp_workdir: str,
    ) -> None:
        self._session = session
        self._run = run
        self._shell_workdir = shell_workdir
        self._sftp_workdir = sftp_workdir

    @staticmethod
    def _validate_operation_id(operation_id: str) -> str:
        value = operation_id.strip().lower()
        if not _OPERATION_RE.fullmatch(value):
            raise ValueError(f"invalid managed operation id: {operation_id!r}")
        return value

    def _operation_dir(self, operation_id: str) -> str:
        safe = self._validate_operation_id(operation_id)
        return f"{self._shell_workdir}/.polaris/operations/{safe}"

    def _attempt_prefix(self, operation_id: str, attempt_id: str) -> str:
        safe_attempt = str(uuid.UUID(attempt_id))
        return f"{self._operation_dir(operation_id)}/attempts/{safe_attempt}"

    def _sftp_attempt_prefix(self, operation_id: str, attempt_id: str) -> str:
        safe = self._validate_operation_id(operation_id)
        safe_attempt = str(uuid.UUID(attempt_id))
        return f"{self._sftp_workdir}/.polaris/operations/{safe}/attempts/{safe_attempt}"

    @staticmethod
    def _pid_from_output(value: object) -> int | None:
        for line in reversed(str(value or "").splitlines()):
            try:
                pid = int(line.strip())
            except ValueError:
                continue
            return pid if pid > 0 else None
        return None

    async def _recover_started_handle(
        self,
        *,
        operation_id: str,
        attempt_id: str,
        context: OperationContext,
        fallback_output: object = "",
    ) -> ManagedCommandHandle | None:
        """Recover when SSH times out after the detached process actually started."""
        prefix = self._attempt_prefix(operation_id, attempt_id)
        fallback_pid = self._pid_from_output(fallback_output)
        pid: int | None = None
        pgid: int | None = None
        for _ in range(10):
            with contextlib.suppress(Exception):
                pid = await self._read_int(f"{prefix}.pid")
                if pid is not None:
                    pgid = await self._read_int(f"{prefix}.pgid") or pid
                    break
            await asyncio.sleep(0.1)
        pid = pid or fallback_pid
        if pid is None:
            return None
        return ManagedCommandHandle(
            operation_id=operation_id,
            attempt_id=attempt_id,
            context=context,
            process_id=pid,
            process_group_id=pgid or pid,
        )

    async def start(
        self,
        context: OperationContext,
        command: str,
        *,
        attempt_id: str | None = None,
    ) -> ManagedCommandHandle:
        operation_id = self._validate_operation_id(context.operation)
        attempt_id = str(uuid.UUID(attempt_id)) if attempt_id else str(uuid.uuid4())
        operation_dir = self._operation_dir(operation_id)
        prefix = self._attempt_prefix(operation_id, attempt_id)
        sftp_prefix = self._sftp_attempt_prefix(operation_id, attempt_id)
        try:
            await self._run(f"mkdir -p {operation_dir}/attempts", 60)
        except Exception as exc:
            raise ManagedCommandLaunchError(context, command, exc) from exc
        lock = f"{operation_dir}/launch.lock"
        try:
            acquired = await self._run(
                f"i=0; until mkdir {lock} 2>/dev/null; do "
                f"if [ -d {lock} ] && [ $(( $(date +%s) - $(stat -c %Y {lock}) )) -gt 120 ]; "
                f"then rmdir {lock} 2>/dev/null || true; fi; "
                "i=$((i+1)); [ $i -lt 240 ] || exit 75; sleep 0.25; done",
                65,
            )
        except Exception as exc:
            raise ManagedCommandLaunchError(context, command, exc) from exc
        if acquired.exit_status != 0:
            cause = RuntimeError("timed out waiting for managed command launch lock")
            raise ManagedCommandLaunchError(context, command, cause)
        try:
            # A concurrent worker may have launched this logical operation while we
            # waited. Attach to it instead of starting a duplicate remote process.
            current = await self.current_attempt_id(operation_id)
            if current is not None:
                current_prefix = self._attempt_prefix(operation_id, current)
                current_pid = await self._read_int(f"{current_prefix}.pid")
                current_exit = await self._read_int(f"{current_prefix}.exit")
                if current_pid is not None and current_exit is None:
                    alive = await self._run(f"kill -0 {current_pid} 2>/dev/null", 60)
                    if alive.exit_status == 0:
                        current_pgid = (
                            await self._read_int(f"{current_prefix}.pgid") or current_pid
                        )
                        return ManagedCommandHandle(
                            operation_id=operation_id,
                            attempt_id=current,
                            context=context,
                            process_id=current_pid,
                            process_group_id=current_pgid,
                        )

            launcher = (
                "#!/usr/bin/env bash\n"
                "set +e\n"
                f"prefix={prefix}\n"
                "echo $$ > ${prefix}.pid\n"
                "ps -o pgid= -p $$ | tr -d ' ' > ${prefix}.pgid\n"
                "date +%s > ${prefix}.started\n"
                "touch ${prefix}.stdout ${prefix}.stderr\n"
                f"{{ {command}; }} >${{prefix}}.stdout 2>${{prefix}}.stderr\n"
                "status=$?\n"
                "printf '%s\\n' \"$status\" > ${prefix}.exit.tmp\n"
                "mv ${prefix}.exit.tmp ${prefix}.exit\n"
                "exit $status\n"
            )
            await self._session.write_file(f"{sftp_prefix}.sh", launcher)
            pointer = f"{operation_dir}/current"
            await self._run(
                f"printf '%s\\n' {attempt_id} > {pointer}.tmp && mv {pointer}.tmp {pointer}",
                60,
            )
            launch_command = (
                f"chmod 700 {prefix}.sh || exit $?; "
                f"nohup setsid bash {prefix}.sh >/dev/null 2>&1 < /dev/null & "
                "pid=$!; printf '%s\\n' \"$pid\""
            )
            try:
                launch = await self._run(launch_command, 60)
            except Exception as exc:
                recovered = await self._recover_started_handle(
                    operation_id=operation_id,
                    attempt_id=attempt_id,
                    context=context,
                    fallback_output=getattr(exc, "stdout", ""),
                )
                if recovered is not None:
                    return recovered
                raise ManagedCommandLaunchError(context, command, exc) from exc
            if launch.exit_status != 0:
                recovered = await self._recover_started_handle(
                    operation_id=operation_id,
                    attempt_id=attempt_id,
                    context=context,
                    fallback_output=launch.stdout,
                )
                if recovered is not None:
                    return recovered
                cause = RuntimeError(redact_text(launch.stderr or launch.stdout, tail=True))
                raise ManagedCommandLaunchError(context, command, cause)
            pid = self._pid_from_output(launch.stdout)
            if pid is None:
                recovered = await self._recover_started_handle(
                    operation_id=operation_id,
                    attempt_id=attempt_id,
                    context=context,
                    fallback_output=launch.stdout,
                )
                if recovered is not None:
                    return recovered
                cause = RuntimeError(
                    f"managed command launch returned no PID: {redact_text(launch.stdout)!r}"
                )
                raise ManagedCommandLaunchError(context, command, cause)
            # The launcher records the real PGID. It normally equals the returned
            # PID; use the PID as a safe initial value until the first snapshot.
            return ManagedCommandHandle(
                operation_id=operation_id,
                attempt_id=attempt_id,
                context=context,
                process_id=pid,
                process_group_id=pid,
            )
        except ManagedCommandLaunchError:
            raise
        except Exception as exc:
            raise ManagedCommandLaunchError(context, command, exc) from exc
        finally:
            with contextlib.suppress(Exception):
                await self._run(f"rmdir {lock} 2>/dev/null || true", 60)

    async def current_attempt_id(self, operation_id: str) -> str | None:
        path = f"{self._operation_dir(operation_id)}/current"
        result = await self._run(f"cat {path} 2>/dev/null", 60)
        if result.exit_status != 0:
            return None
        try:
            return str(uuid.UUID(result.stdout.strip().splitlines()[-1]))
        except (ValueError, IndexError):
            return None

    async def _read_int(self, path: str) -> int | None:
        result = await self._run(f"cat {path} 2>/dev/null", 60)
        if result.exit_status != 0:
            return None
        try:
            return int(result.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            return None

    async def snapshot(
        self,
        handle: ManagedCommandHandle,
        *,
        previous_token: str | None = None,
        diagnostic_evidence: dict[str, str] | None = None,
    ) -> CommandSnapshot:
        prefix = self._attempt_prefix(handle.operation_id, handle.attempt_id)
        pid = await self._read_int(f"{prefix}.pid") or handle.process_id
        pgid = await self._read_int(f"{prefix}.pgid") or handle.process_group_id
        started = await self._read_int(f"{prefix}.started") or int(time.time())
        exit_status = await self._read_int(f"{prefix}.exit")
        alive_result = await self._run(f"kill -0 {pid} 2>/dev/null", 60)
        alive = alive_result.exit_status == 0
        stat_result = await self._run(
            f"stat -c '%s %Y' {prefix}.stdout {prefix}.stderr 2>/dev/null",
            60,
        )
        sizes: list[int] = []
        mtimes: list[int] = []
        for line in stat_result.stdout.splitlines():
            try:
                size_text, mtime_text = line.split()[-2:]
                sizes.append(int(size_text))
                mtimes.append(int(mtime_text))
            except (ValueError, IndexError):
                continue
        stdout = await self._run(f"tail -c 8000 {prefix}.stdout 2>/dev/null", 60)
        stderr = await self._run(f"tail -c 8000 {prefix}.stderr 2>/dev/null", 60)
        process = await self._run(
            f"ps -o etimes=,time=,stat=,wchan= -p {pid} 2>/dev/null",
            60,
        )
        cpu_seconds: float | None = None
        process_state: str | None = None
        if process.stdout.strip():
            fields = process.stdout.strip().split()
            if len(fields) >= 3:
                process_state = " ".join(fields[2:])
                try:
                    parts = [int(part) for part in fields[1].split(":")]
                    cpu_seconds = float(
                        sum(
                            value * (60**index)
                            for index, value in enumerate(reversed(parts))
                        )
                    )
                except ValueError:
                    pass
        now = time.time()
        snapshot = CommandSnapshot(
            operation_id=handle.operation_id,
            attempt_id=handle.attempt_id,
            context=handle.context,
            elapsed_seconds=max(0.0, now - started),
            process_alive=alive,
            exit_status=exit_status,
            stdout_tail=stdout.stdout if stdout.exit_status == 0 else "",
            stderr_tail=stderr.stdout if stderr.exit_status == 0 else "",
            output_bytes=sum(sizes),
            seconds_since_output=max(0.0, now - max(mtimes or [started])),
            cpu_seconds=cpu_seconds,
            process_state=process_state,
            process_id=pid,
            process_group_id=pgid,
            diagnostic_evidence=diagnostic_evidence,
        )
        snapshot.output_changed = (
            previous_token is not None and snapshot.progress_token != previous_token
        )
        return snapshot

    async def read_output(
        self,
        handle: ManagedCommandHandle,
        *,
        stdout_offset: int = 0,
        stderr_offset: int = 0,
    ) -> tuple[list[OutputChunk], int, int]:
        prefix = self._attempt_prefix(handle.operation_id, handle.attempt_id)
        chunks: list[OutputChunk] = []
        next_offsets: list[int] = []
        for stream, offset in (("stdout", stdout_offset), ("stderr", stderr_offset)):
            safe_offset = max(0, int(offset))
            result = await self._run(
                f"tail -c +{safe_offset + 1} {prefix}.{stream} 2>/dev/null",
                60,
            )
            text = result.stdout if result.exit_status == 0 else ""
            next_offset = safe_offset + len(text.encode("utf-8"))
            next_offsets.append(next_offset)
            if text:
                chunks.append(OutputChunk(stream=stream, text=text, offset=next_offset))
        return chunks, next_offsets[0], next_offsets[1]

    async def diagnose(self, handle: ManagedCommandHandle) -> dict[str, str]:
        pid = int(handle.process_id)
        probes = {
            "process_tree": (
                f"ps -o pid,ppid,pgid,stat,etime,time,wchan:32,args -p {pid} --no-headers "
                f"2>/dev/null; ps --ppid {pid} -o pid,ppid,pgid,stat,etime,time,args "
                "--no-headers 2>/dev/null"
            ),
            "resources": "LC_ALL=C free -m 2>/dev/null; df -Pk . 2>/dev/null",
            "network": f"ss -tpn 2>/dev/null | grep -F 'pid={pid},' | head -n 20",
        }
        evidence: dict[str, str] = {}
        for name, command in probes.items():
            result = await self._run(command, 60)
            evidence[name] = redact_text(result.stdout or result.stderr or "unavailable", tail=True)
        return evidence

    async def stop(self, handle: ManagedCommandHandle) -> bool:
        current = await self.current_attempt_id(handle.operation_id)
        if current != handle.attempt_id:
            return False
        snapshot = await self.snapshot(handle)
        if not snapshot.process_alive:
            return True
        pgid = int(snapshot.process_group_id or handle.process_group_id)
        await self._run(
            f"kill -TERM -- -{pgid} 2>/dev/null || true; "
            f"i=0; while kill -0 {snapshot.process_id} 2>/dev/null && [ $i -lt 20 ]; "
            "do sleep 0.25; i=$((i+1)); done; "
            f"kill -KILL -- -{pgid} 2>/dev/null || true",
            60,
        )
        verified = await self._run(f"kill -0 {snapshot.process_id} 2>/dev/null", 60)
        return verified.exit_status != 0
