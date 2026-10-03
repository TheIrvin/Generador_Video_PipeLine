from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from animavideo import main


def _png_image() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (8, 8), color="red").save(buffer, format="PNG")
    return buffer.getvalue()


def test_health_endpoint_returns_service_status():
    response = TestClient(main.app).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "generador-video-pipeline"}


def test_invalid_render_option_type_returns_422_before_creating_job(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "ROOT", tmp_path)
    client = TestClient(main.app, raise_server_exceptions=False)

    response = client.post(
        "/v1/jobs",
        files={"image": ("source.png", _png_image(), "image/png")},
        data={"prompt": "Preserve the source image", "options": '{"fps":"24"}'},
    )

    assert response.status_code == 422
    assert "fps" in response.json()["detail"]
    assert list(tmp_path.iterdir()) == []


def test_valid_upload_creates_queued_job_without_running_renderer(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "ROOT", tmp_path)
    monkeypatch.setattr(main, "_run_job", lambda _job_id: None)
    client = TestClient(main.app)

    response = client.post(
        "/v1/jobs",
        files={"image": ("source.png", _png_image(), "image/png")},
        data={"prompt": "Preserve the source image"},
    )

    assert response.status_code == 202
    job = response.json()
    assert job["status"] == "queued"
    assert client.get(job["status_url"]).json()["original_filename"] == "source.png"
    assert list(tmp_path.iterdir()) == [tmp_path / job["job_id"]]
