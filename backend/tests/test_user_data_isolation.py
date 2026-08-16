from pathlib import Path
import uuid

from fastapi.testclient import TestClient

from app.api.candidates import _delete_cv_files
from app.main import create_app


def _create_owner_account(client: TestClient, label: str) -> dict[str, str]:
    email = f"{label}-{uuid.uuid4().hex[:8]}@example.com"
    password = "strongpass123"
    response = client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": password, "full_name": label.title()},
    )
    assert response.status_code == 201, response.text
    assert response.json()["role"] == "owner"

    login = client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": password},
    )
    assert login.status_code == 200, login.text
    return {"Authorization": f"Bearer {login.json()['access_token']}"}


def test_owner_data_is_isolated_by_account() -> None:
    app = create_app()
    fixture = Path(__file__).parent / "fixtures" / "cv_engineer.txt"
    cv_text = fixture.read_text(encoding="utf-8")
    marker = uuid.uuid4().hex[:8]
    candidate_email = f"isolated-{marker}@example.com"
    cv_a = cv_text.replace("Alex Engineer", f"Isolated Alpha {marker}").replace(
        "alex.engineer@example.com", candidate_email
    ).encode()
    cv_b = cv_text.replace("Alex Engineer", f"Isolated Beta {marker}").replace(
        "alex.engineer@example.com", candidate_email
    ).encode()

    with TestClient(app) as client:
        owner_a = _create_owner_account(client, "owner-a")
        owner_b = _create_owner_account(client, "owner-b")

        job_a = client.post(
            "/api/v1/jobs",
            json={"description": "Python backend engineer with FastAPI."},
            headers=owner_a,
        )
        job_b = client.post(
            "/api/v1/jobs",
            json={"description": "JavaScript frontend engineer with React."},
            headers=owner_b,
        )
        assert job_a.status_code == 200, job_a.text
        assert job_b.status_code == 200, job_b.text
        job_a_id = job_a.json()["job_id"]
        job_b_id = job_b.json()["job_id"]

        candidate_a = client.post(
            "/api/v1/candidates?use_llm=false",
            files={"file": ("candidate-a.txt", cv_a, "text/plain")},
            headers=owner_a,
        )
        candidate_b = client.post(
            "/api/v1/candidates?use_llm=false",
            files={"file": ("candidate-b.txt", cv_b, "text/plain")},
            headers=owner_b,
        )
        assert candidate_a.status_code == 200, candidate_a.text
        assert candidate_b.status_code == 200, candidate_b.text
        candidate_a_id = candidate_a.json()["candidate_id"]
        candidate_b_id = candidate_b.json()["candidate_id"]
        assert candidate_a_id != candidate_b_id

        _delete_cv_files(candidate_a_id)
        downloaded_cv = client.get(
            f"/api/v1/candidates/{candidate_a_id}/cv?download=true",
            headers=owner_a,
        )
        assert downloaded_cv.status_code == 200
        assert downloaded_cv.content == cv_a
        assert "attachment;" in downloaded_cv.headers["content-disposition"]

        jobs_a = {row["job_id"] for row in client.get("/api/v1/jobs", headers=owner_a).json()}
        jobs_b = {row["job_id"] for row in client.get("/api/v1/jobs", headers=owner_b).json()}
        candidates_a = {row["candidate_id"] for row in client.get("/api/v1/candidates", headers=owner_a).json()}
        candidates_b = {row["candidate_id"] for row in client.get("/api/v1/candidates", headers=owner_b).json()}
        assert job_a_id in jobs_a and job_b_id not in jobs_a
        assert job_b_id in jobs_b and job_a_id not in jobs_b
        assert candidate_a_id in candidates_a and candidate_b_id not in candidates_a
        assert candidate_b_id in candidates_b and candidate_a_id not in candidates_b

        assert client.get(f"/api/v1/candidates/{candidate_b_id}", headers=owner_a).status_code == 404
        assert client.get(f"/api/v1/candidates/{candidate_a_id}/cv", headers=owner_b).status_code == 404
        assert client.patch(
            f"/api/v1/jobs/{job_b_id}",
            json={"title": "Must remain isolated"},
            headers=owner_a,
        ).status_code == 404
        assert client.delete(f"/api/v1/jobs/{job_a_id}", headers=owner_b).status_code == 404

        matching = client.post(
            f"/api/v1/jobs/{job_a_id}/match?use_hybrid=false&top_k=100",
            headers=owner_a,
        )
        assert matching.status_code == 200, matching.text
        matching_ids = {row["candidate_id"] for row in matching.json()["results"]}
        assert candidate_a_id in matching_ids
        assert candidate_b_id not in matching_ids

        cross_report = client.post(
            "/api/v1/reports/compare",
            json={"job_id": job_a_id, "candidate_ids": [candidate_b_id]},
            headers=owner_a,
        )
        assert cross_report.status_code == 404

        cross_feedback = client.post(
            "/api/v1/matching/feedback",
            json={"job_id": job_a_id, "candidate_id": candidate_b_id, "skill": "python"},
            headers=owner_a,
        )
        assert cross_feedback.status_code == 404

        cross_interview = client.post(
            "/api/v1/interviews/start?use_llm=false",
            json={"job_id": job_a_id, "candidate_id": candidate_a_id},
            headers=owner_b,
        )
        assert cross_interview.status_code == 404

        assert client.delete(f"/api/v1/jobs/{job_b_id}", headers=owner_b).status_code == 200
