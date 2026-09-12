from transformers import TrainerControl, TrainerState

from xslm.callbacks import TimeLimitCallback


def test_the_run_continues_before_the_limit():
    callback = TimeLimitCallback(time_limit_minutes=60)
    control = TrainerControl()
    callback.on_step_end(args=None, state=TrainerState(), control=control)
    assert control.should_training_stop is False


def test_the_run_stops_after_the_limit():
    callback = TimeLimitCallback(time_limit_minutes=0)
    control = TrainerControl()
    callback.on_step_end(args=None, state=TrainerState(), control=control)
    assert control.should_training_stop is True
    assert control.should_save is True


def test_the_clock_starts_at_the_beginning_of_training():
    callback = TimeLimitCallback(time_limit_minutes=60)
    callback.on_train_begin(args=None, state=TrainerState(), control=TrainerControl())
    assert callback.started_at is not None
