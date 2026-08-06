from configs import load_config


def test_load_mean_reversion_config():
    config = load_config("mean_reversion")
    assert config["strategy"] == "mean_reversion"
    assert "params" in config
    assert "costs" in config
    assert "risk" in config
