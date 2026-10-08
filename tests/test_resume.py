"""Real optimizer/scheduler/RNG continuation on a tiny synthetic model."""
import random
import pytest

torch = pytest.importorskip('torch')
np = pytest.importorskip('numpy')
from train import save_checkpoint, restore_checkpoint


def test_epoch_boundary_resume_matches_uninterrupted_update(tmp_path):
    torch.manual_seed(3)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3)
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    config = {'fold': 0, 'patch': 64, 'world_size': 1, 'effective_batch_size': 1}

    def update():
        optimizer.zero_grad()
        loss = model(torch.randn(4, 3)).square().mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        return float(loss.detach())

    update()
    save_checkpoint(tmp_path / 'last.pt', model, optimizer, scheduler, scaler,
                    epoch=1, best=.2, history=[{'epoch': 1}], config=config,
                    split_id='split', rank_states=None)
    expected_loss = update()
    expected = {k: v.clone() for k, v in model.state_dict().items()}
    expected_random = (random.random(), np.random.rand())
    state = restore_checkpoint(tmp_path / 'last.pt', model, optimizer, scheduler,
                               scaler, config, 'split', rank=0)
    assert state['epoch'] == 1
    assert update() == expected_loss
    assert (random.random(), np.random.rand()) == expected_random
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, expected[k], rtol=0, atol=0)
    with pytest.raises(ValueError, match='incompatible'):
        restore_checkpoint(tmp_path / 'last.pt', model, optimizer, scheduler, scaler,
                           {**config, 'world_size': 2}, 'split', rank=0)

    with pytest.raises(ValueError, match='incompatible'):
        restore_checkpoint(tmp_path / 'last.pt', model, optimizer, scheduler, scaler,
                           {**config, 'fold': 1}, 'split', rank=0)
