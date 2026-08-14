from app.services.managed_commands import (
    CommandAction,
    CommandSnapshot,
    CommandState,
    ModelAssessment,
    OperationContext,
    RecoveryPlan,
    RepairScope,
    adjudicate_command,
    failure_from_snapshot,
    may_apply_recovery_automatically,
)


def _snapshot(**overrides):
    values = {
        "operation_id": "setup",
        "attempt_id": "attempt-1",
        "context": OperationContext(
            phase="dependency.install",
            operation="install dependencies",
            display_command="pip install -r requirements.txt",
            target="host",
            soft_timeout_seconds=60,
            stall_timeout_seconds=120,
            repair_scope=RepairScope.DEPENDENCY_FILES,
        ),
        "elapsed_seconds": 180,
        "process_alive": True,
        "exit_status": None,
        "stdout_tail": "downloading package",
        "output_bytes": 20,
        "seconds_since_output": 180,
        "process_id": 123,
        "process_group_id": 123,
    }
    values.update(overrides)
    return CommandSnapshot(**values)


def test_progress_is_stronger_than_model_stall_guess():
    snap = _snapshot(output_changed=True)
    advice = ModelAssessment(
        state=CommandState.STALLED,
        confidence=0.99,
        reason="guess",
        proposed_action=CommandAction.STOP_AND_REPAIR,
        safe_to_interrupt=True,
    )
    assert adjudicate_command(snap, advice).action == CommandAction.CONTINUE_MONITORING


def test_unknown_at_stall_threshold_asks_without_stopping():
    advice = ModelAssessment(
        state=CommandState.UNKNOWN,
        confidence=0.2,
        reason="insufficient evidence",
    )
    verdict = adjudicate_command(_snapshot(), advice)
    assert verdict.action == CommandAction.ASK_USER_WHILE_RUNNING


def test_missing_model_fails_safe_and_eventually_asks():
    before = _snapshot(seconds_since_output=30)
    stalled = _snapshot(seconds_since_output=180)
    assert adjudicate_command(before, None).action == CommandAction.CONTINUE_MONITORING
    assert adjudicate_command(stalled, None).action == CommandAction.ASK_USER_WHILE_RUNNING


def test_slow_operation_gets_one_extension_then_user_review():
    advice = ModelAssessment(
        state=CommandState.SLOW,
        confidence=0.9,
        reason="healthy but slow",
        proposed_action=CommandAction.EXTEND_OBSERVATION,
    )
    assert adjudicate_command(_snapshot(), advice).action == CommandAction.EXTEND_OBSERVATION
    assert (
        adjudicate_command(_snapshot(), advice, silent_extensions=1).action
        == CommandAction.ASK_USER_WHILE_RUNNING
    )


def test_high_confidence_stall_requires_diagnostic_before_stop():
    advice = ModelAssessment(
        state=CommandState.STALLED,
        confidence=0.95,
        reason="process is blocked",
        proposed_action=CommandAction.STOP_AND_REPAIR,
        safe_to_interrupt=True,
    )
    assert adjudicate_command(_snapshot(), advice).action == CommandAction.RUN_DIAGNOSTIC
    assert (
        adjudicate_command(_snapshot(), advice, diagnostics_run=1).action
        == CommandAction.STOP_AND_REPAIR
    )


def test_nonzero_exit_returns_real_failure_evidence_and_redacts_secrets():
    snap = _snapshot(
        process_alive=False,
        exit_status=1,
        stderr_tail="token=secret-value\nHTTP 500",
    )
    verdict = adjudicate_command(snap, None)
    report = failure_from_snapshot(snap)
    assert verdict.action == CommandAction.REPORT_FAILURE
    assert report.exit_status == 1
    assert report.stderr_tail is not None
    assert "secret-value" not in report.stderr_tail
    assert report.repair_scope == RepairScope.DEPENDENCY_FILES


def test_automatic_repair_is_confidence_scope_and_progress_bounded():
    safe = RecoveryPlan(
        diagnosis="invalid generated dependency pin",
        confidence=0.9,
        repair_scope=RepairScope.DEPENDENCY_FILES,
        proposed_changes=("requirements.txt",),
        expected_evidence="pip exits zero",
        minimal_retry="dependency.install",
    )
    assert may_apply_recovery_automatically(safe, repeated_without_progress=0)
    assert not may_apply_recovery_automatically(safe, repeated_without_progress=2)
    infrastructure = RecoveryPlan(
        diagnosis="registry unavailable",
        confidence=0.99,
        repair_scope=RepairScope.INFRASTRUCTURE,
        proposed_changes=("requirements.txt",),
        expected_evidence="registry responds",
        minimal_retry="artifact.download",
    )
    assert not may_apply_recovery_automatically(
        infrastructure, repeated_without_progress=0
    )
