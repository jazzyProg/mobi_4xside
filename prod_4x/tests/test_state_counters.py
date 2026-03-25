from __future__ import annotations


def test_new_detail_resets_per_detail_counters():
    from app.core.state import QCEngineState

    s = QCEngineState()
    s.start_new_detail()
    s.register_detail_result(True)

    assert s.stats["current_detail"]["checked"] == 1
    assert s.stats["current_detail"]["pass_percent"] == 100.0

    s.start_new_detail()
    assert s.stats["current_detail"]["checked"] == 0
    assert s.stats["current_detail"]["passed"] == 0
    assert s.stats["current_detail"]["failed"] == 0


def test_shift_counter_keeps_running_when_new_detail_starts():
    from app.core.state import QCEngineState

    s = QCEngineState()
    s.start_new_detail()
    s.register_detail_result(True)
    s.start_new_detail()
    s.register_detail_result(False)

    assert s.stats["shift"]["checked_total"] == 2
    assert s.stats["shift"]["passed_total"] == 1
    assert s.stats["shift"]["failed_total"] == 1
