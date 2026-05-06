from vn_receipt_ocr.train.manifest import build_manifest


def test_manifest_required_fields():
    m = build_manifest(
        config_dict={"a": 1},
        run_id="abc",
        seed=42,
        gpu_device_name="Tesla P100",
    )
    assert m["run_id"] == "abc"
    assert m["seed"] == 42
    assert m["gpu_device_name"] == "Tesla P100"
    assert "python_version" in m
    assert "library_versions" in m
    assert m["config"] == {"a": 1}
    assert "git_commit" in m  # may be empty string if not in a repo
