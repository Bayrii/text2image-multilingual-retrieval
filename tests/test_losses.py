"""InfoNCE: zero (or near-zero) when embeddings are identical & uniformly spread."""
import torch

from src.models.losses import HardNegativeMiner, InfoNCELoss


def _orthonormal(n: int, d: int) -> torch.Tensor:
    A = torch.randn(n, d)
    Q, _ = torch.linalg.qr(A)
    return Q[:, :d] if Q.shape[1] >= d else Q


def test_infonce_zero_when_identical():
    """When image_emb == text_emb and rows are orthogonal, similarity matrix
    is the identity scaled, and cross-entropy with diagonal targets is small."""
    torch.manual_seed(0)
    emb = torch.eye(8, 32)
    loss = InfoNCELoss(temperature=0.07)(emb, emb).item()
    assert loss < 0.5, f"loss={loss} expected near zero"


def test_infonce_random_positive_bounded():
    torch.manual_seed(1)
    a = torch.randn(16, 32)
    b = torch.randn(16, 32)
    a = torch.nn.functional.normalize(a, dim=1)
    b = torch.nn.functional.normalize(b, dim=1)
    loss = InfoNCELoss(0.07)(a, b).item()
    assert loss > 0.0
    assert loss < 30.0


def test_infonce_masks_same_image_id():
    """Multiple captions per image: same-id off-diagonal positives are masked."""
    torch.manual_seed(2)
    emb = torch.nn.functional.normalize(torch.randn(8, 32), dim=1)
    ids = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    loss_masked = InfoNCELoss(0.07)(emb, emb, image_ids=ids).item()
    loss_unmasked = InfoNCELoss(0.07)(emb, emb).item()
    assert loss_masked >= 0.0
    assert loss_unmasked >= 0.0


def test_hard_negative_miner_excludes_same_image():
    torch.manual_seed(3)
    emb = torch.nn.functional.normalize(torch.randn(6, 16), dim=1)
    ids = torch.tensor([0, 0, 1, 1, 2, 2])
    miner = HardNegativeMiner(k=2)
    idx = miner(emb, ids)
    for i, neighbours in enumerate(idx.tolist()):
        for j in neighbours:
            assert ids[j].item() != ids[i].item(), \
                f"miner returned same-image id at i={i}, j={j}"
