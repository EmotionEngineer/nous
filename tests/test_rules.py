import torch
from nous.rules import FixedPairRuleLayer, SoftmaxRuleLayer, SparseRuleLayer, SimpleNousBlock

def test_fixed_pair_rule_layer():
    layer = FixedPairRuleLayer(input_dim=8, num_rules=5)
    facts = torch.rand(3, 8)
    out = layer(facts)
    assert out.shape == (3, 5)
    assert torch.isfinite(out).all()

def test_simple_block_forward():
    blk = SimpleNousBlock(in_dim=8, n_rules=6)
    x = torch.rand(2, 8)
    y = blk(x)
    assert y.shape == (2, 6)

def test_softmax_rule_layer():
    layer = SoftmaxRuleLayer(input_dim=10, num_rules=7, top_k_facts=3, top_k_rules=5)
    x = torch.rand(4, 10)
    y = layer(x)
    assert y.shape == (4, 7)

def test_sparse_rule_layer_train_eval():
    layer = SparseRuleLayer(input_dim=6, num_rules=4)
    x = torch.rand(2, 6)
    layer.train()
    y = layer(x)
    assert y.shape == (2, 4)
    layer.eval()
    y2 = layer(x)
    assert y2.shape == (2, 4)
    l0 = layer.compute_l0_loss()
    assert l0 >= 0.0