from vn_receipt_ocr.kaggle.secrets import get_secret_or_none


def test_returns_none_when_no_kaggle(monkeypatch):
    # Simulate: not on Kaggle, no UserSecretsClient
    monkeypatch.setattr(
        "vn_receipt_ocr.kaggle.secrets._client_factory",
        lambda: None,
    )
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    assert get_secret_or_none("WANDB_API_KEY") is None


def test_returns_env_var_when_present(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "abc123")
    assert get_secret_or_none("WANDB_API_KEY") == "abc123"
