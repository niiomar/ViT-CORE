from train import make_lr_lambda, should_stop_early


def test_lr_lambda_warmup_ramps_linearly():
    lr_lambda = make_lr_lambda(warmup_epochs=4, total_epochs=20, lr=1e-4, min_lr=1e-6)
    assert lr_lambda(0) == 0.25
    assert lr_lambda(3) == 1.0


def test_lr_lambda_decays_to_min_lr_floor_at_final_epoch():
    lr, min_lr = 1e-4, 1e-6
    lr_lambda = make_lr_lambda(warmup_epochs=4, total_epochs=20, lr=lr, min_lr=min_lr)
    # cosine term hits its floor once progress reaches 1.0, i.e. at (and past) the last epoch.
    assert abs(lr_lambda(20) - min_lr / lr) < 1e-9


def test_should_stop_early_respects_patience():
    assert should_stop_early(epochs_since_improvement=10, patience=10) is True
    assert should_stop_early(epochs_since_improvement=9, patience=10) is False


def test_should_stop_early_disabled_when_patience_not_positive():
    assert should_stop_early(epochs_since_improvement=1000, patience=0) is False
    assert should_stop_early(epochs_since_improvement=1000, patience=-1) is False
