# tests/test_checkpoint.py

import torch

import config.reinforce as REINFORCE_CONFIG
from src.algorithms.reinforce import PolicyNetwork
from src.checkpoint import (
    checkpoint_path,
    save_checkpoint,
    should_print_progress,
    should_save_checkpoint,
)


# Test the initial untrained policy (update 0) is checkpointed
def test_checkpoint_at_update_zero():
    assert should_save_checkpoint(0, REINFORCE_CONFIG.CHECKPOINT_INTERVAL_UPDATES)


# Test checkpoints follow the configured update interval
def test_checkpoint_uses_configured_interval():
    interval = REINFORCE_CONFIG.CHECKPOINT_INTERVAL_UPDATES

    assert not should_save_checkpoint(interval - 1, REINFORCE_CONFIG.CHECKPOINT_INTERVAL_UPDATES)
    assert should_save_checkpoint(interval, REINFORCE_CONFIG.CHECKPOINT_INTERVAL_UPDATES)
    assert should_save_checkpoint(2 * interval, REINFORCE_CONFIG.CHECKPOINT_INTERVAL_UPDATES)


# Test checkpoint timing is independent of batch size
def test_checkpoint_timing_independent_of_batch_size(monkeypatch):
    baseline = [should_save_checkpoint(update, REINFORCE_CONFIG.CHECKPOINT_INTERVAL_UPDATES) for update in range(401)]

    for batch_size in (1, 7, 20, 33, 1000):
        monkeypatch.setattr(REINFORCE_CONFIG, "BATCH_SIZE", batch_size)
        assert [should_save_checkpoint(update, REINFORCE_CONFIG.CHECKPOINT_INTERVAL_UPDATES) for update in range(401)] == baseline


# Test progress printing follows its independently configured update interval
def test_progress_printing_uses_configured_interval():
    interval = REINFORCE_CONFIG.PRINT_INTERVAL_UPDATES

    assert should_print_progress(0, REINFORCE_CONFIG.PRINT_INTERVAL_UPDATES)
    assert not should_print_progress(interval - 1, REINFORCE_CONFIG.PRINT_INTERVAL_UPDATES)
    assert should_print_progress(interval, REINFORCE_CONFIG.PRINT_INTERVAL_UPDATES)
    assert should_print_progress(2 * interval, REINFORCE_CONFIG.PRINT_INTERVAL_UPDATES)


# Test filenames use the _update_<N>.pt suffix
def test_checkpoint_filenames_use_updates(tmp_path):
    timestamp = "2026-09-19_160023"

    assert checkpoint_path(tmp_path, timestamp, 0).name == f"{timestamp}_update_0.pt"
    assert checkpoint_path(tmp_path, timestamp, 100).name == f"{timestamp}_update_100.pt"
    assert checkpoint_path(tmp_path, timestamp, 200).name == f"{timestamp}_update_200.pt"


# Test saved metadata records completed updates and sampled episodes
def test_checkpoint_metadata(tmp_path):
    policy = PolicyNetwork()
    optimiser = torch.optim.Adam(policy.parameters(), lr=REINFORCE_CONFIG.LEARNING_RATE)

    # 100 updates of batch size 20 is 2000 sampled episodes, but the two are stored separately
    path = save_checkpoint(policy, optimiser, 100, 2000, tmp_path)
    checkpoint = torch.load(path, map_location="cpu")

    assert path.parent == tmp_path
    assert path.name.endswith("_update_100.pt")
    assert checkpoint["update"] == 100
    assert checkpoint["episodes_sampled"] == 2000
    assert "episode" not in checkpoint
    assert set(checkpoint["policy_state_dict"]) == set(policy.state_dict())
    assert "optimiser_state_dict" in checkpoint


# Test the update 0 checkpoint holds the untrained policy
def test_initial_checkpoint_holds_untrained_policy(tmp_path):
    policy = PolicyNetwork()
    optimiser = torch.optim.Adam(policy.parameters(), lr=REINFORCE_CONFIG.LEARNING_RATE)

    path = save_checkpoint(policy, optimiser, 0, 0, tmp_path)
    checkpoint = torch.load(path, map_location="cpu")

    assert path.name.endswith("_update_0.pt")
    assert checkpoint["update"] == 0
    assert checkpoint["episodes_sampled"] == 0

    for name, tensor in policy.state_dict().items():
        assert torch.equal(checkpoint["policy_state_dict"][name], tensor)
