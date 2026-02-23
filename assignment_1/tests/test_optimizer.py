import numpy
import torch

from .adapters import get_adamw_cls, run_get_lr_cosine_schedule, run_muon_on_quadratic
from basic.model import Muon


def _optimize(opt_class) -> torch.Tensor:
    torch.manual_seed(42)
    model = torch.nn.Linear(3, 2, bias=False)
    opt = opt_class(
        model.parameters(),
        lr=1e-3,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    # Use 1000 optimization steps for testing
    for _ in range(1000):
        opt.zero_grad()
        x = torch.rand(model.in_features)
        y_hat = model(x)
        y = torch.tensor([x[0] + x[1], -x[2]])
        loss = ((y - y_hat) ** 2).sum()
        loss.backward()
        opt.step()
    return model.weight.detach()


def test_adamw(numpy_snapshot):
    """
    Our reference implementation yields slightly different results than the
    PyTorch AdamW, since there are a couple different ways that you can apply
    weight decay that are equivalent in principle, but differ in practice due to
    floating point behavior. So, we test that the provided implementation matches
    _either_ our reference implementation's expected results or those from the PyTorch AdamW.
    """
    # expected_weights = torch.load(FIXTURES_PATH / "adamw_expected_params.pt")
    pytorch_weights = _optimize(torch.optim.AdamW)
    actual_weights = _optimize(get_adamw_cls())

    # Might need to exit early if the weights match pytorch, since that should also be valid
    matches_pytorch = torch.allclose(actual_weights, pytorch_weights, atol=1e-4)
    if matches_pytorch:
        return

    numpy_snapshot.assert_match(
        actual_weights,
        atol=1e-4,
    )


def test_get_lr_cosine_schedule():
    max_learning_rate = 1
    min_learning_rate = 1 * 0.1
    warmup_iters = 7
    cosine_cycle_iters = 21

    expected_lrs = [
        0,
        0.14285714285714285,
        0.2857142857142857,
        0.42857142857142855,
        0.5714285714285714,
        0.7142857142857143,
        0.8571428571428571,
        1.0,
        0.9887175604818206,
        0.9554359905560885,
        0.9018241671106134,
        0.8305704108364301,
        0.7452476826029011,
        0.6501344202803414,
        0.55,
        0.44986557971965857,
        0.3547523173970989,
        0.26942958916356996,
        0.19817583288938662,
        0.14456400944391146,
        0.11128243951817937,
        0.1,
        0.1,
        0.1,
        0.1,
    ]
    actual_lrs = [
        run_get_lr_cosine_schedule(
            it=it,
            max_learning_rate=max_learning_rate,
            min_learning_rate=min_learning_rate,
            warmup_iters=warmup_iters,
            cosine_cycle_iters=cosine_cycle_iters,
        )
        for it in range(25)
    ]
    numpy.testing.assert_allclose(numpy.array(actual_lrs), numpy.array(expected_lrs))


def test_muon_reduces_quadratic_loss():
    initial_loss, final_loss = run_muon_on_quadratic()
    assert final_loss < initial_loss


def test_muon_step_with_no_grad_is_noop():
    param = torch.nn.Parameter(torch.tensor([[1.0, -2.0]], dtype=torch.float32))
    before = param.detach().clone()
    opt = Muon(
        [param],
        lr=1e-2,
        weight_decay=0.0,
        momentum=0.95,
        a=3.4445,
        b=-4.7750,
        c=2.0315,
        eps=1e-8,
        cautious_decay=False,
    )

    param.grad = None
    opt.step()

    torch.testing.assert_close(param.detach(), before)


def test_muon_momentum_buffer_tracks_raw_ema():
    param = torch.nn.Parameter(torch.tensor([[1.0, -2.0], [0.5, 3.0]], dtype=torch.float32))
    grad = torch.tensor([[0.3, -0.7], [0.2, 0.1]], dtype=torch.float32)
    opt = Muon(
        [param],
        lr=1e-2,
        weight_decay=0.0,
        momentum=0.9,
        a=3.4445,
        b=-4.7750,
        c=2.0315,
        eps=1e-8,
        cautious_decay=False,
    )

    param.grad = grad.clone()
    opt.step()
    state = opt.state[param]
    torch.testing.assert_close(state["momentum_matrix"], grad)

    param.grad = grad.clone()
    opt.step()
    expected = grad * 1.9  # 0.9 * grad + grad
    torch.testing.assert_close(state["momentum_matrix"], expected)
