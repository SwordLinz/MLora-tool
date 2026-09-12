from app import runninghub_batch_workflow as rh


def test_output_extension_accepts_latent_file_type():
    item = {"fileType": "latent", "fileUrl": "https://example.test/output?id=1"}

    assert rh.output_extension(item) == "latent"


def test_output_extension_accepts_latent_url_suffix():
    item = {
        "fileType": "",
        "fileUrl": "https://example.test/files/SHires_232721_00001_psuba_1781969352.latent?download=1",
    }

    assert rh.output_extension(item) == "latent"


def test_output_extension_normalizes_jpeg():
    item = {"fileType": "jpeg", "fileUrl": "https://example.test/output"}

    assert rh.output_extension(item) == "jpg"
