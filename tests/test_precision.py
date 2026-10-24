"""Forward overflow retries must preserve model buffers, RNG and accumulated gradients."""
from copy import deepcopy
import pytest
torch = pytest.importorskip('torch')
from scripts import training_runtime as runtime


def test_forward_retry_matches_one_float32_pass_and_keeps_accumulated_gradients():
    class OverflowModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = torch.nn.BatchNorm3d(2)
            self.dropout = torch.nn.Dropout(.5)
            self.conv = torch.nn.Conv3d(2, 2, 1)
            self.modes = []

        def forward(self, x):
            amp = torch.is_autocast_enabled(x.device.type)
            self.modes.append(amp)
            result = self.conv(self.dropout(self.norm(x)))
            if amp:
                # A failed convolution can poison BatchNorm state downstream.
                self.norm.running_mean.fill_(float('nan'))
                result = result * float('nan')
            return result

    torch.manual_seed(4)
    model = OverflowModel().train()
    reference = deepcopy(model)
    inputs = torch.randn(1, 2, 4, 4, 4)
    labels = torch.zeros_like(inputs)
    for p in list(model.parameters()) + list(reference.parameters()):
        p.grad = torch.ones_like(p)  # Earlier microbatch contributions must survive.
    loss_fn = torch.nn.MSELoss()
    torch.manual_seed(12)
    expected = loss_fn(reference(inputs), labels)
    expected.backward()
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(12)
    loss, retried = runtime.training_loss(model, inputs, labels, loss_fn, amp_enabled=True)
    loss.backward()
    assert retried and model.modes == [True, False]
    torch.testing.assert_close(loss, expected, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, reference.state_dict()[key], rtol=0, atol=0)
    for actual, expected_param in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual.grad, expected_param.grad, rtol=0, atol=0)


def test_bad_input_and_persistent_nonfinite_loss_are_not_hidden():
    model = torch.nn.Sequential(torch.nn.BatchNorm3d(2), torch.nn.Conv3d(2, 2, 1)).train()
    inputs = torch.randn(1, 2, 4, 4, 4)
    labels = torch.zeros_like(inputs)
    with pytest.raises(RuntimeError, match='Nonfinite training input'):
        runtime.training_loss(model, inputs * float('nan'), labels, torch.nn.MSELoss(), amp_enabled=True)
    initial = deepcopy(model.state_dict())
    with pytest.raises(RuntimeError, match='Nonfinite training loss.*float32'):
        runtime.training_loss(model, inputs, labels, lambda logits, labels: logits.sum() * float('nan'), amp_enabled=True)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, initial[key], rtol=0, atol=0)


def test_inference_retries_nonfinite_amp_output_and_rejects_invalid_results():
    class OverflowModel(torch.nn.Module):
        def forward(self, inputs):
            return inputs * (float('nan') if torch.is_autocast_enabled(inputs.device.type) else 2)

    inputs = torch.ones(1, 2, 4, 4, 4)
    actual = runtime.inference_forward(OverflowModel(), inputs, amp_enabled=True)
    torch.testing.assert_close(actual, inputs * 2)
    with pytest.raises(RuntimeError, match='Nonfinite inference input'):
        runtime.inference_forward(OverflowModel(), inputs * float('nan'), amp_enabled=True)
    with pytest.raises(RuntimeError, match='Nonfinite inference output'):
        runtime.inference_forward(lambda x: x * float('nan'), inputs, amp_enabled=True)
