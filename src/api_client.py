import os
import time


class PostureIngestClient:
    def __init__(
        self,
        ingest_url: str | None = None,
        device_id: str | None = None,
        secret: str | None = None,
        timeout_s: float = 5.0,
    ):
        self.ingest_url = ingest_url or os.environ.get(
            'POSTURE_INGEST_URL',
            os.environ.get(
                'POSTURE_API_URL',
                'https://posturepal-atm8oc0g8-theox.vercel.app/api/posture',
            ),
        )
        self.device_id = device_id or os.environ.get('DEVICE_ID', 'pi-posture-001')
        self.secret = secret or os.environ.get('POSTURE_API_SECRET') or os.environ.get('UPDATE_SECRET')
        self.timeout_s = timeout_s

    def send_posture_record(
        self,
        posture_type: str,
        confidence: float,
        metrics: dict | None = None,
        keypoints: list | None = None,
        retries: int = 2,
    ) -> bool:
        try:
            import requests
        except Exception as e:
            print(f"[API] requests not available: {e}")
            return False

        payload = {
            'device_id': self.device_id,
            'posture_type': posture_type,
            'confidence': float(confidence),
            'metrics': metrics or {},
            'keypoints': keypoints or {},
        }

        headers = {'Content-Type': 'application/json'}
        if self.secret:
            headers['x-posture-secret'] = self.secret

        last_err = None
        for attempt in range(retries + 1):
            try:
                resp = requests.post(
                    self.ingest_url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout_s,
                )
                if 200 <= resp.status_code < 300:
                    return True
                last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
            except Exception as e:
                last_err = str(e)

            if attempt < retries:
                time.sleep(0.5 * (attempt + 1))

        print(f"[API] upload failed: {last_err}")
        return False
