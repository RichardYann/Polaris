"""Runner 抽象的单元测试：现有 SSH 执行器就是 RemoteHostRunner，且满足 kind 无关的 Runner 接口。

外加 ContainerRunner：执行原语应在容器内跑（docker exec 包裹），文件原语走 host 侧；
以及 container 规格的严格白名单校验（防注入）与 open_runner 的声明式分派。"""

import uuid
from unittest.mock import AsyncMock

import pytest

from app.agents.voyage import runner
from app.services import ssh_exec
from app.services.managed_commands import CommandSnapshot, OperationContext, RepairScope

# Runner 接口应覆盖的 kind 无关原语（实验循环只依赖这些）。
_PRIMITIVES = (
    "workdir",
    "mkdir_workdir",
    "write_files",
    "read_file",
    "list_dir",
    "read_metrics_json",
    "setup_venv",
    "launch_setup",
    "read_setup_exit",
    "read_setup_log",
    "run_smoke",
    "run_plot",
    "ensure_plot_deps",
    "launch_run",
    "check_pid",
    "read_exit_code",
    "tail_log",
    "kill_pid",
    "close",
)


def test_remote_host_runner_is_current_ssh_executor():
    """现有行为 = RemoteHostRunner（SSH 主机上的裸机 venv 执行）——零行为变化的解耦。"""
    assert runner.RemoteHostRunner is ssh_exec.SSHExecutor


def test_runner_protocol_covers_primitives():
    for name in _PRIMITIVES:
        assert hasattr(runner.Runner, name), f"Runner 协议缺原语：{name}"


def test_ssh_executor_satisfies_runner():
    """现有 SSHExecutor 结构上满足 Runner（未来 Container/Api Runner 挂同一接口）。"""
    for name in _PRIMITIVES:
        assert hasattr(ssh_exec.SSHExecutor, name), f"SSHExecutor 缺原语：{name}"


def test_container_runner_is_runner_subclass():
    """ContainerRunner 复用同一接口（继承 SSHExecutor，故文件原语等全部满足 Runner）。"""
    assert issubclass(runner.ContainerRunner, ssh_exec.SSHExecutor)
    for name in _PRIMITIVES:
        assert hasattr(runner.ContainerRunner, name), f"ContainerRunner 缺原语：{name}"


# ---- container 规格白名单校验（安全边界：值会拼进 docker 命令） ----


def test_parse_container_spec_valid_and_defaults():
    spec = runner.parse_container_spec(
        {"image": "verlai/verl:vllm017.latest", "gpus": "device=0,1"}
    )
    assert spec is not None
    assert spec.image == "verlai/verl:vllm017.latest"
    assert spec.gpus == "device=0,1"
    assert spec.gpu_mode == "auto"
    assert spec.shm_size == "16g"  # 默认
    assert spec.mounts == {"~/hf": "/hf:ro"}  # 默认挂载
    assert spec.workdir_mount == "/work"


def test_parse_container_spec_missing_image_is_none():
    """无 image / 非 dict → None（=退回裸机，不用容器）。"""
    assert runner.parse_container_spec(None) is None
    assert runner.parse_container_spec({}) is None
    assert runner.parse_container_spec({"gpus": "all"}) is None
    assert runner.parse_container_spec("verl") is None


@pytest.mark.parametrize(
    "bad_image",
    ["evil; rm -rf /", "img$(whoami)", "a b", "img`id`", "img'x", 'img"x'],
)
def test_parse_container_spec_rejects_injection_in_image(bad_image):
    """非法 image（含 shell 元字符/空格/引号）整体拒绝——绝不进 docker 命令。"""
    assert runner.parse_container_spec({"image": bad_image}) is None


def test_parse_container_spec_drops_bad_gpus_and_mounts():
    """image 合法但 gpus/mounts 非法 → 丢弃该字段回退默认，而非拒绝整个 spec。"""
    spec = runner.parse_container_spec(
        {
            "image": "myimg:1",
            "gpus": "device=0; reboot",  # 非法 → 丢弃
            "shm_size": "16g; rm",  # 非法 → 回退默认
            "mounts": {"/data": "/data", "/bad;x": "/y"},  # 后者非法 → 只留合法项
        }
    )
    assert spec is not None
    assert spec.gpus is None
    assert spec.shm_size == "16g"
    assert spec.mounts == {"/data": "/data"}  # 非法挂载被剔除


@pytest.mark.parametrize("mode", ["auto", "gpus", "nvidia_runtime"])
def test_parse_container_spec_accepts_gpu_modes(mode):
    spec = runner.parse_container_spec(
        {"image": "myimg:1", "gpus": "2", "gpu_mode": mode}
    )
    assert spec is not None
    assert spec.gpu_mode == mode


def test_parse_container_spec_invalid_gpu_mode_falls_back_to_auto():
    spec = runner.parse_container_spec(
        {"image": "myimg:1", "gpus": "2", "gpu_mode": "$(id)"}
    )
    assert spec is not None
    assert spec.gpu_mode == "auto"


@pytest.mark.parametrize("count", ["0", "257", "999999"])
def test_parse_container_spec_drops_unsafe_gpu_counts(count):
    spec = runner.parse_container_spec({"image": "myimg:1", "gpus": count})
    assert spec is not None
    assert spec.gpus is None


# ---- docker exec 命令拼装（纯字符串构造，不触 SSH/DB） ----


def _container_runner(
    gpus="device=2,3", image="verlai/verl:vllm017.latest", gpu_mode="auto"
):
    exp_id = str(uuid.uuid4())
    spec = runner.ContainerSpec(image=image, gpus=gpus, gpu_mode=gpu_mode)
    return runner.ContainerRunner(
        object(),  # 仅测字符串构造，不调用 run，session 用不到
        exp_id=exp_id,
        host="gpu.example",
        project_id=uuid.uuid4(),
        spec=spec,
    )


def test_docker_run_cmd_has_gpus_mounts_and_workdir():
    r = _container_runner(gpus="device=2,3")
    cmd = r._docker_run_cmd()
    assert cmd.startswith("docker run -d")
    assert f"--name polaris_{r.exp_id}" in cmd
    assert "--gpus '\"device=2,3\"'" in cmd  # 多卡需引号形式
    assert "--shm-size 16g" in cmd
    assert "-v ~/hf:/hf:ro" in cmd
    assert f"-v {r.workdir}:/work" in cmd  # host workdir ←→ /work
    assert cmd.endswith("-w /work verlai/verl:vllm017.latest tail -f /dev/null")


def test_docker_run_cmd_gpus_all_and_count():
    assert "--gpus all" in _container_runner(gpus="all")._docker_run_cmd()
    assert "--gpus 4" in _container_runner(gpus="4")._docker_run_cmd()


def test_docker_run_cmd_nvidia_runtime_uses_visible_devices():
    cmd = _container_runner(gpus="device=2,3", gpu_mode="nvidia_runtime")._docker_run_cmd()
    assert "--runtime=nvidia" in cmd
    assert "-e NVIDIA_VISIBLE_DEVICES=2,3" in cmd
    assert "-e NVIDIA_DRIVER_CAPABILITIES=all" in cmd
    assert "--gpus" not in cmd


def test_nvidia_runtime_converts_gpu_count_to_device_indices():
    cmd = _container_runner(gpus="4", gpu_mode="nvidia_runtime")._docker_run_cmd()
    assert "NVIDIA_VISIBLE_DEVICES=0,1,2,3" in cmd
    assert "NVIDIA_VISIBLE_DEVICES=4" not in cmd


@pytest.mark.parametrize(
    "text",
    [
        "unknown flag: --gpus",
        'could not select device driver "" with capabilities: [[gpu]]',
        (
            "Auto-detected mode as 'legacy'\n"
            "nvidia-container-cli: mount error: failed to add device rules: "
            "write /sys/fs/cgroup/devices/docker/id/devices.allow: operation not permitted"
        ),
    ],
)
def test_gpu_runtime_errors_require_legacy_fallback(text):
    assert runner.ContainerRunner._gpu_error_requires_legacy(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "dial tcp registry-1.docker.io: i/o timeout",
        "unexpected commit digest sha256:abc, expected sha256:def",
        "no space left on device",
        "mount error: operation not permitted",
        "Conflict. The container name is already in use",
    ],
)
def test_unrelated_docker_errors_do_not_require_legacy_fallback(text):
    assert runner.ContainerRunner._gpu_error_requires_legacy(text) is False


def _prepare_snapshot(*, phase="environment.prepare.gpus", stderr=""):
    return CommandSnapshot(
        operation_id="environment-prepare",
        attempt_id=str(uuid.uuid4()),
        context=OperationContext(
            phase=phase,
            operation="environment-prepare",
            display_command="start container using --gpus",
            target="gpu.example",
            repair_scope=RepairScope.INFRASTRUCTURE,
        ),
        elapsed_seconds=1,
        process_alive=False,
        exit_status=125,
        stderr_tail=stderr,
    )


@pytest.mark.asyncio
async def test_managed_prepare_falls_back_once_for_confirmed_gpu_runtime_error():
    r = _container_runner(gpus="2", gpu_mode="auto")
    r._has_nvidia_runtime = AsyncMock(return_value=True)
    r._run = AsyncMock(return_value=ssh_exec.SSHResult(0, "", ""))
    expected = object()
    r.start_managed_command = AsyncMock(return_value=expected)

    result = await r.recover_managed_prepare(
        _prepare_snapshot(
            stderr=(
                "nvidia-container-cli: mount error: failed to add device rules: "
                "write devices.allow: operation not permitted"
            )
        )
    )

    assert result is expected
    context, command = r.start_managed_command.await_args.args
    assert context.phase == "environment.prepare.nvidia_runtime"
    assert "--runtime=nvidia" in command
    assert "NVIDIA_VISIBLE_DEVICES=0,1" in command

    second = await r.recover_managed_prepare(
        _prepare_snapshot(
            phase="environment.prepare.nvidia_runtime",
            stderr="same failure",
        )
    )
    assert second is None
    assert r.start_managed_command.await_count == 1


@pytest.mark.asyncio
async def test_managed_prepare_does_not_fallback_without_registered_runtime():
    r = _container_runner(gpus="all", gpu_mode="auto")
    r._has_nvidia_runtime = AsyncMock(return_value=False)
    r.start_managed_command = AsyncMock()

    result = await r.recover_managed_prepare(
        _prepare_snapshot(stderr="unknown flag: --gpus")
    )

    assert result is None
    r.start_managed_command.assert_not_awaited()


@pytest.mark.asyncio
async def test_sync_prepare_retries_legacy_runtime_after_confirmed_failure():
    r = _container_runner(gpus="2", gpu_mode="auto")
    r._run = AsyncMock(
        side_effect=[
            ssh_exec.SSHResult(1, "", ""),  # container is not running
            ssh_exec.SSHResult(0, "", ""),  # remove stale container
            ssh_exec.SSHResult(
                125,
                "",
                "nvidia-container-cli: failed to add device rules: "
                "devices.allow: operation not permitted",
            ),
            ssh_exec.SSHResult(0, '{"nvidia":{"path":"nvidia-container-runtime"}}', ""),
            ssh_exec.SSHResult(0, "", ""),  # remove failed container
            ssh_exec.SSHResult(0, "container-id", ""),
        ]
    )

    await r._ensure_container()

    commands = [call.args[0] for call in r._run.await_args_list]
    assert sum(command.startswith("docker run -d") for command in commands) == 2
    assert any("--gpus 2" in command for command in commands)
    assert any("--runtime=nvidia" in command for command in commands)


def test_dexec_wraps_in_container_and_cds_to_workdir():
    r = _container_runner()
    inner = r._dexec_workdir("bash run.sh --smoke")
    assert inner == f"docker exec polaris_{r.exp_id} bash -lc 'cd /work && bash run.sh --smoke'"


def test_dexec_rejects_single_quote_to_avoid_injection():
    r = _container_runner()
    with pytest.raises(ssh_exec.SSHExecError):
        r._dexec("echo 'oops'")
