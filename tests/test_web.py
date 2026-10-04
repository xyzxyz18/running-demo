import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import app as web_app


class WebAppTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.jobs_patch = patch.object(web_app, "JOBS_DIR", Path(self.temp_dir.name))
        self.jobs_patch.start()
        web_app.jobs.clear()
        self.client = web_app.app.test_client()

    def tearDown(self):
        self.jobs_patch.stop()
        self.temp_dir.cleanup()

    def test_home_page_loads(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"videoInput", response.data)
        self.assertIn(b"exportPdfButton", response.data)

    def test_pdf_artifact_can_be_downloaded(self):
        job_id = "pdf-job"
        folder = Path(self.temp_dir.name) / job_id
        folder.mkdir()
        (folder / "report.pdf").write_bytes(b"%PDF-1.4\n")
        web_app.jobs[job_id] = {"id": job_id, "state": "completed"}
        response = self.client.get(f"/results/{job_id}/report.pdf")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "application/pdf")
        response.close()

    def test_rejects_missing_or_invalid_video(self):
        self.assertEqual(self.client.post("/api/jobs").status_code, 400)
        response = self.client.post(
            "/api/jobs",
            data={"video": (io.BytesIO(b"not a video"), "sample.txt")},
        )
        self.assertEqual(response.status_code, 400)

    def test_realtime_endpoint_requires_jpeg(self):
        response = self.client.post("/api/realtime/pose", data=b"frame")
        self.assertEqual(response.status_code, 415)

    @patch.object(web_app.executor, "submit")
    def test_accepts_video_and_queues_job(self, submit):
        response = self.client.post(
            "/api/jobs",
            data={"video": (io.BytesIO(b"video bytes"), "sample.mp4")},
        )
        self.assertEqual(response.status_code, 202)
        job_id = response.get_json()["id"]
        self.assertEqual(self.client.get(f"/api/jobs/{job_id}").get_json()["state"], "queued")
        self.assertEqual(web_app.jobs[job_id]['model'], 'rtmpose')
        history = self.client.get("/api/jobs").get_json()["jobs"]
        self.assertEqual([item["id"] for item in history], [job_id])
        self.assertTrue(any((Path(self.temp_dir.name) / job_id).glob("*.mp4")))
        source_response = self.client.get(f"/results/{job_id}/source.mp4")
        self.assertEqual(source_response.status_code, 200)
        source_response.close()
        submit.assert_called_once()

    @patch.object(web_app.executor, "submit")
    def test_reanalyzes_from_preserved_source(self, submit):
        created = self.client.post(
            "/api/jobs",
            data={"video": (io.BytesIO(b"video bytes"), "sample.mp4")},
        ).get_json()
        response = self.client.post(f"/api/jobs/{created['id']}/reanalyze")
        self.assertEqual(response.status_code, 202)
        self.assertNotEqual(response.get_json()["id"], created["id"])
        self.assertEqual(submit.call_count, 2)

    def test_deletes_completed_history_and_files(self):
        job_id = "delete-job"
        folder = Path(self.temp_dir.name) / job_id
        folder.mkdir()
        (folder / "job.json").write_text("{}", encoding="utf-8")
        web_app.jobs[job_id] = {"id": job_id, "state": "completed"}

        response = self.client.delete(f"/api/jobs/{job_id}")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.get_json()["deleted"])
        self.assertNotIn(job_id, web_app.jobs)
        self.assertFalse(folder.exists())

    def test_cannot_delete_running_history(self):
        job_id = "running-job"
        folder = Path(self.temp_dir.name) / job_id
        folder.mkdir()
        web_app.jobs[job_id] = {"id": job_id, "state": "running"}

        response = self.client.delete(f"/api/jobs/{job_id}")

        self.assertEqual(response.status_code, 409)
        self.assertTrue(folder.exists())

    @patch.object(web_app.executor, 'submit')
    def test_model_is_queued_persisted_and_preserved_on_reload(self, submit):
        response = self.client.post('/api/jobs', data={
            'video': (io.BytesIO(b'video'), 'test.mp4'), 'model': 'rtmpose'})
        job_id = response.get_json()['id']
        self.assertEqual(web_app.jobs[job_id]['model'], 'rtmpose')
        metadata = json.loads((web_app.JOBS_DIR / job_id / 'job.json').read_text())
        self.assertEqual(metadata['model'], 'rtmpose')
        web_app.jobs.clear()
        web_app.load_existing_jobs()
        self.assertEqual(web_app.jobs[job_id]['model'], 'rtmpose')
        response = self.client.post(f'/api/jobs/{job_id}/reanalyze', json={})
        new_id = response.get_json()['id']
        self.assertEqual(web_app.jobs[new_id]['model'], 'rtmpose')
        self.assertEqual(web_app.jobs[job_id]['model'], 'rtmpose')

    @patch.object(web_app.executor, 'submit')
    def test_old_model_history_is_reanalyzed_with_rtmpose(self, submit):
        created = self.client.post('/api/jobs', data={
            'video': (io.BytesIO(b'video'), 'test.mp4')}).get_json()
        old_id = created['id']
        web_app.jobs[old_id]['model'] = 'mediapipe'
        response = self.client.post(f'/api/jobs/{old_id}/reanalyze')
        self.assertEqual(response.status_code, 202)
        self.assertEqual(web_app.jobs[response.get_json()['id']]['model'], 'rtmpose')
        self.assertEqual(web_app.jobs[old_id]['model'], 'mediapipe')

    @patch.object(web_app.executor, 'submit')
    def test_invalid_model_does_not_create_job(self, submit):
        response = self.client.post('/api/jobs', data={
            'video': (io.BytesIO(b'video'), 'test.mp4'), 'model': 'invalid'})
        self.assertEqual(response.status_code, 400)
        self.assertFalse(web_app.jobs)
        submit.assert_not_called()

    @patch.object(web_app.realtime_pose, 'process', return_value={'detected': False})
    def test_realtime_forwards_selected_model(self, process):
        response = self.client.post('/api/realtime/pose', data=b'jpeg', content_type='image/jpeg')
        self.assertEqual(response.status_code, 200)
        process.assert_called_once_with(b'jpeg', 'rtmpose')


if __name__ == "__main__":
    unittest.main()
