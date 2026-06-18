"""Verify image-id-level split has no leakage."""
from src.data.dataset import split_by_image_id


def _make_rows(n_images: int, captions_per_image: int = 5):
    rows = []
    for iid in range(n_images):
        for c in range(captions_per_image):
            rows.append({"image_id": iid, "caption": f"img{iid}-cap{c}",
                         "file_path": f"/img/{iid}.jpg", "language": "en"})
    return rows


def test_split_no_leakage():
    rows = _make_rows(200, 5)
    train, val, test = split_by_image_id(rows, val_split=0.1, test_split=0.1, seed=0)
    train_ids = {r["image_id"] for r in train}
    val_ids = {r["image_id"] for r in val}
    test_ids = {r["image_id"] for r in test}
    assert not (train_ids & val_ids)
    assert not (train_ids & test_ids)
    assert not (val_ids & test_ids)
    assert (len(train_ids) + len(val_ids) + len(test_ids)) == 200
    assert len(train) + len(val) + len(test) == len(rows)


def test_split_proportions():
    rows = _make_rows(1000, 5)
    train, val, test = split_by_image_id(rows, val_split=0.1, test_split=0.1, seed=1)
    assert 90 <= len({r["image_id"] for r in val}) <= 110
    assert 90 <= len({r["image_id"] for r in test}) <= 110


def test_split_deterministic():
    rows = _make_rows(50, 2)
    a1, a2, a3 = split_by_image_id(rows, 0.2, 0.2, seed=42)
    b1, b2, b3 = split_by_image_id(rows, 0.2, 0.2, seed=42)
    assert {r["image_id"] for r in a1} == {r["image_id"] for r in b1}
    assert {r["image_id"] for r in a2} == {r["image_id"] for r in b2}
    assert {r["image_id"] for r in a3} == {r["image_id"] for r in b3}
