import base64
import json
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from elderly_monitoring.modules.roi_annotation.ezviz_client import EzvizVisionModelResult
from elderly_monitoring.modules.roi_annotation.service import annotate_roi_image
from elderly_monitoring.modules.roi_annotation.validation import RoiValidationError, parse_model_json, validate_roi_payload
from elderly_monitoring.service.app import create_app
from elderly_monitoring.service.settings import ServiceSettings


PNG_1X1 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="


def valid_payload() -> dict:
    return {
        "schema_version": "roi_annotation_v1",
        "rois": [
            {
                "roi_id": "roi_sofa_01",
                "type": "sofa",
                "shape": {
                    "type": "polygon",
                    "points": [
                        {"x": 0.1, "y": 0.2},
                        {"x": 0.4, "y": 0.2},
                        {"x": 0.4, "y": 0.6},
                    ],
                    "bbox": {"x1": 0.1, "y1": 0.2, "x2": 0.4, "y2": 0.6},
                },
                "confidence": 0.4,
            }
        ],
    }


class FakeRoiClient:
    def annotate_rois(self, **kwargs):
        self.kwargs = kwargs
        return EzvizVisionModelResult(
            content=json.dumps(valid_payload(), ensure_ascii=False),
            model="qwen3.6-plus",
            request_id="mock_request",
            usage={"total_tokens": 1},
            raw_response={"mock": True},
        )


class RoiValidationTest(unittest.TestCase):
    def test_parse_markdown_wrapped_json_and_defaults(self) -> None:
        parsed = parse_model_json("```json\n" + json.dumps(valid_payload()) + "\n```")
        normalized = validate_roi_payload(parsed, image_width=658, image_height=419, expected_types=["sofa", "doorway"])
        self.assertEqual(normalized["rois"][0]["type"], "sofa")
        self.assertFalse(normalized["rois"][0]["activity_stats"]["allowed"])
        self.assertIn("low_confidence", normalized["rois"][0]["quality_flags"])
        self.assertEqual(normalized["missing_expected"], ["doorway"])

    def test_rejects_bad_geometry(self) -> None:
        payload = valid_payload()
        payload["rois"][0]["shape"]["bbox"]["x1"] = 0.8
        with self.assertRaisesRegex(RoiValidationError, "bbox 范围无效"):
            validate_roi_payload(payload, image_width=658, image_height=419)

    def test_rejects_unknown_type(self) -> None:
        payload = valid_payload()
        payload["rois"][0]["type"] = "sink"
        with self.assertRaisesRegex(RoiValidationError, "未知 ROI type"):
            validate_roi_payload(payload, image_width=658, image_height=419)

    def test_annotate_roi_image_hashes_without_returning_base64(self) -> None:
        client = FakeRoiClient()
        result = annotate_roi_image(
            image_base64=PNG_1X1,
            mime_type="image/png",
            image_width=1,
            image_height=1,
            scene_hint="客厅",
            expected_types=["sofa"],
            client=client,
        )
        text = json.dumps(result, ensure_ascii=False)
        self.assertIn("sha256", result["image"])
        self.assertNotIn(PNG_1X1, text)
        self.assertEqual(client.kwargs["mime_type"], "image/jpeg")


class RoiServiceApiTest(unittest.TestCase):
    def setUp(self) -> None:
        settings = ServiceSettings(api_token="api", model_path="missing.pt", ezviz_llm_api_key="fake-key")
        self.client = TestClient(create_app(settings=settings))

    def headers(self) -> dict[str, str]:
        return {"Authorization": "Bearer api"}

    def test_requires_auth(self) -> None:
        response = self.client.post("/v1/roi/annotate", json={})
        self.assertEqual(response.status_code, 401)

    def test_validates_request_shape(self) -> None:
        response = self.client.post(
            "/v1/roi/annotate",
            headers=self.headers(),
            json={"image_base64": "x", "mime_type": "image/gif", "image_width": 1, "image_height": 1},
        )
        self.assertEqual(response.status_code, 422)

    def test_annotate_endpoint_returns_structured_roi(self) -> None:
        with patch("elderly_monitoring.service.app.EzvizVisionModelClient", return_value=FakeRoiClient()):
            response = self.client.post(
                "/v1/roi/annotate",
                headers=self.headers(),
                json={
                    "image_base64": PNG_1X1,
                    "mime_type": "image/png",
                    "image_width": 1,
                    "image_height": 1,
                    "scene_hint": "客厅",
                    "expected_types": ["sofa", "doorway"],
                },
            )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["schema_version"], "roi_annotation_v1")
        self.assertEqual(data["rois"][0]["type"], "sofa")
        self.assertEqual(data["missing_expected"], ["doorway"])
        self.assertIn("sha256", data["image"])


def base64_sha256(value: str) -> str:
    import hashlib

    return hashlib.sha256(base64.b64decode(value)).hexdigest()


if __name__ == "__main__":
    unittest.main()
