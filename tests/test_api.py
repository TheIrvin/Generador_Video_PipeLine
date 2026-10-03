import json
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


def test_startup_marks_interrupted_jobs_failed_and_preserves_terminal_jobs(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "ROOT", tmp_path)
    jobs = {
        "a" * 32: {"job_id": "a" * 32, "status": "queued", "error": None},
        "b" * 32: {
            "job_id": "b" * 32,
            "status": "processing",
            "started_at": "2026-10-03T12:00:00+00:00",
        },
        "c" * 32: {"job_id": "c" * 32, "status": "completed", "result": {"frames": 24}},
        "d" * 32: {"job_id": "d" * 32, "status": "failed", "error": "render error"},
    }
    for job_id, job in jobs.items():
        job_directory = tmp_path / job_id
        job_directory.mkdir()
        (job_directory / "job.json").write_text(json.dumps(job), encoding="utf-8")

    with TestClient(main.app) as client:
        assert client.get("/health").status_code == 200
        for job_id in ("a" * 32, "b" * 32):
            recovered = client.get(f"/v1/jobs/{job_id}")
            assert recovered.status_code == 200
            assert recovered.json()["status"] == "failed"

    for job_id in ("a" * 32, "b" * 32):
        job = json.loads((tmp_path / job_id / "job.json").read_text(encoding="utf-8"))
        assert job["status"] == "failed"
        assert "service restart" in job["error"]
        assert job["completed_at"]

    for job_id in ("c" * 32, "d" * 32):
        job = json.loads((tmp_path / job_id / "job.json").read_text(encoding="utf-8"))
        assert job == jobs[job_id]


def test_startup_skips_unreadable_or_invalid_job_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "ROOT", tmp_path)
    unreadable = tmp_path / ("e" * 32)
    unreadable.mkdir()
    (unreadable / "job.json").write_text("{invalid json", encoding="utf-8")
    invalid_id = tmp_path / "not-a-job-id"
    invalid_id.mkdir()
    (invalid_id / "job.json").write_text(
        json.dumps({"job_id": "not-a-job-id", "status": "processing"}), encoding="utf-8"
    )
    malformed_status = tmp_path / ("f" * 32)
    malformed_status.mkdir()
    (malformed_status / "job.json").write_text(
        json.dumps({"job_id": "f" * 32, "status": []}), encoding="utf-8"
    )

    with TestClient(main.app) as client:
        assert client.get("/health").status_code == 200

    assert (unreadable / "job.json").read_text(encoding="utf-8") == "{invalid json"
    invalid_job = json.loads((invalid_id / "job.json").read_text(encoding="utf-8"))
    assert invalid_job["status"] == "processing"
    malformed_job = json.loads((malformed_status / "job.json").read_text(encoding="utf-8"))
    assert malformed_job["status"] == []
