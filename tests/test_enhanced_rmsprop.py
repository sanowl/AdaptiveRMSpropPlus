import pytest
import torch
import torch.nn as nn

from adaptive_rmsprop.optimizer import EnhancedRMSpropTF

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)
        
    def forward(self, x):
        return self.linear(x)

@pytest.fixture
def setup_optimizer():
    model = SimpleModel()
    optimizer = EnhancedRMSpropTF(
        model.parameters(),
        lr=1e-2,
        alpha=0.99,
        eps=1e-8,
        momentum=0.9,
        noise_scale=1e-6,
        warmup_steps=10,
        lr_cycles=3
    )
    return model, optimizer

@pytest.fixture
def sample_batch():
    X = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    y = torch.tensor([[3.0], [7.0]], requires_grad=True)
    return X, y

def test_optimizer_initialization():
    """Test if optimizer initializes correctly"""
    model = SimpleModel()

    opt = EnhancedRMSpropTF(model.parameters(), lr=1e-3)
    assert opt.defaults["lr"] == 1e-3
    assert opt.defaults["alpha"] == 0.99

    with pytest.raises(ValueError):
        EnhancedRMSpropTF(model.parameters(), lr=-1)

    with pytest.raises(ValueError):
        EnhancedRMSpropTF(model.parameters(), alpha=1.0)

    with pytest.raises(ValueError):
        EnhancedRMSpropTF(model.parameters(), gradient_clip=0)

    with pytest.raises(ValueError):
        EnhancedRMSpropTF(model.parameters(), warmup_steps=-1)

def test_noise_injection():
    """Test noise injection mechanism"""
    torch.manual_seed(42)  # For reproducibility
    
    model = SimpleModel()
    optimizer = EnhancedRMSpropTF(
        model.parameters(),
        lr=0.1,
        noise_scale=0.1,
        eps=1e-8
    )
    
    # Store initial parameters and set gradients
    initial_params = []
    for p in model.parameters():
        initial_params.append(p.clone().detach())
        p.grad = torch.ones_like(p)
    
    # Take multiple optimization steps
    for _ in range(5):
        optimizer.step()
    
    # Check if parameters have changed
    all_stable = True
    all_changed = False
    
    for p, initial_p in zip(model.parameters(), initial_params):
        if torch.isnan(p).any():
            pytest.fail("NaN values detected in parameters")
            
        diff = torch.abs(p - initial_p).mean()
        if diff > 1e-4:
            all_changed = True
        if torch.isinf(diff):
            all_stable = False
    
    assert all_stable, "Parameters contain infinite values"
    assert all_changed, "Parameters should change significantly"

def test_cyclic_learning_rate():
    """Test cyclic learning rate"""
    model = SimpleModel()
    optimizer = EnhancedRMSpropTF(
        model.parameters(),
        lr=0.1,
        warmup_steps=10,
        lr_cycles=2
    )
    
    lr_0 = optimizer._compute_cyclic_lr(0)
    lr_5 = optimizer._compute_cyclic_lr(5)
    assert lr_0 < lr_5


def test_cyclic_learning_rate_supports_floor_and_no_warmup():
    """Test cyclic learning rate floor and no-warmup mode"""
    model = SimpleModel()
    optimizer = EnhancedRMSpropTF(
        model.parameters(),
        lr=0.1,
        warmup_steps=0,
        lr_cycles=2,
        min_lr_ratio=0.25,
    )

    lrs = [optimizer._compute_cyclic_lr(step) for step in range(10)]
    assert max(lrs) <= 0.1
    assert min(lrs) >= 0.025
    assert lrs[0] == pytest.approx(0.1)

def test_epsilon_adjustment():
    """Test epsilon adjustment"""
    model = SimpleModel()
    optimizer = EnhancedRMSpropTF(model.parameters())
    
    small_grad = torch.tensor(0.001)
    large_grad = torch.tensor(1000.0)
    
    eps_small = optimizer._adjust_epsilon(small_grad)
    eps_large = optimizer._adjust_epsilon(large_grad)
    
    assert eps_large > eps_small

def test_training_loop(setup_optimizer, sample_batch):
    """Test training loop"""
    model, optimizer = setup_optimizer
    X, y = sample_batch
    
    initial_loss = None
    
    for epoch in range(20):
        optimizer.zero_grad()
        output = model(X)
        loss = nn.MSELoss()(output, y)
        
        if epoch == 0:
            initial_loss = loss.item()
        
        if torch.isnan(loss):
            pytest.fail(f"NaN loss detected at epoch {epoch}")
            
        loss.backward()
        
        for p in model.parameters():
            if torch.isnan(p.grad).any():
                pytest.fail(f"NaN gradients detected at epoch {epoch}")
        
        optimizer.step()
        
        if loss.item() < initial_loss * 0.9:
            break
    
    final_loss = loss.item()
    assert not torch.isnan(torch.tensor(final_loss)), "Final loss is NaN"
    assert final_loss < initial_loss, f"Loss should decrease. Initial: {initial_loss}, Final: {final_loss}"

def test_momentum_buffer():
    """Test momentum buffer"""
    model = SimpleModel()
    optimizer = EnhancedRMSpropTF(
        model.parameters(),
        momentum=0.9
    )
    
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    
    optimizer.step()
    
    for group in optimizer.param_groups:
        for p in group['params']:
            state = optimizer.state[p]
            assert 'momentum_buffer' in state


def test_gradient_clip_limits_optimizer_statistics():
    """Test that gradient clipping is applied before square-average update"""
    param = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = EnhancedRMSpropTF(
        [param],
        lr=0.1,
        alpha=0.0,
        eps=1e-8,
        noise_scale=0.0,
        warmup_steps=0,
        gradient_clip=2.0,
    )
    param.grad = torch.tensor([10.0])

    optimizer.step()

    state = optimizer.state[param]
    assert state["square_avg"].item() == pytest.approx(4.0)


def test_decoupled_weight_decay_with_zero_gradient():
    """Test AdamW-style weight decay path"""
    param = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = EnhancedRMSpropTF(
        [param],
        lr=0.1,
        alpha=0.0,
        eps=1e-8,
        weight_decay=0.1,
        noise_scale=0.0,
        warmup_steps=0,
        decoupled_weight_decay=True,
    )
    param.grad = torch.zeros_like(param)

    optimizer.step()

    assert param.item() == pytest.approx(0.99)


def test_maximize_moves_parameter_up_gradient():
    """Test maximize mode reverses update direction"""
    param = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = EnhancedRMSpropTF(
        [param],
        lr=0.1,
        alpha=0.0,
        eps=1e-8,
        noise_scale=0.0,
        warmup_steps=0,
        maximize=True,
    )
    param.grad = torch.ones_like(param)

    optimizer.step()

    assert param.item() > 1.0


def test_parameter_groups_use_their_own_learning_rates():
    """Test per-group learning rates are respected by the scheduler"""
    first = torch.nn.Parameter(torch.tensor([1.0]))
    second = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = EnhancedRMSpropTF(
        [
            {"params": [first], "lr": 0.1},
            {"params": [second], "lr": 0.01},
        ],
        alpha=0.0,
        eps=1e-8,
        noise_scale=0.0,
        warmup_steps=0,
    )
    first.grad = torch.ones_like(first)
    second.grad = torch.ones_like(second)

    optimizer.step()

    first_delta = 1.0 - first.item()
    second_delta = 1.0 - second.item()
    assert first_delta > second_delta
    assert first_delta / second_delta == pytest.approx(10.0, rel=1e-4)

def test_state_dict():
    """Test state dict save/load"""
    model = SimpleModel()
    optimizer = EnhancedRMSpropTF(model.parameters())
    
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    
    optimizer.step()
    state_dict = optimizer.state_dict()
    
    new_optimizer = EnhancedRMSpropTF(model.parameters())
    new_optimizer.load_state_dict(state_dict)
    
    assert optimizer.state.keys() == new_optimizer.state.keys()

if __name__ == "__main__":
    pytest.main([__file__]) 