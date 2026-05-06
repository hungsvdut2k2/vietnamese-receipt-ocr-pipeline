from vn_receipt_ocr.kaggle.gpu_detect import detect_gpu_profile_name


def test_detect_p100():
    assert detect_gpu_profile_name(device_name="Tesla P100-PCIE-16GB") == "p100_16gb"


def test_detect_t4():
    assert detect_gpu_profile_name(device_name="Tesla T4") == "t4_16gb"


def test_detect_l4():
    assert detect_gpu_profile_name(device_name="NVIDIA L4") == "l4_24gb"


def test_detect_unknown_returns_default():
    assert detect_gpu_profile_name(device_name="weird gpu", default="p100_16gb") == "p100_16gb"
