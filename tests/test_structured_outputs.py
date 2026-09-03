import json
import unittest

from pydantic import ValidationError

from scripts import structured_outputs as so


class StructuredOutputsTest(unittest.TestCase):
    def test_task_models_accept_valid_probability_confidence(self):
        cases = [
            (
                so.Task1Response,
                {"decision": "yes", "confidence": 0.0, "brief_reason": "explicit must"},
            ),
            (
                so.Task2Response,
                {
                    "requirement": "The system MAY export reports.",
                    "modality": "optional",
                    "confidence": 0.95,
                },
            ),
            (
                so.Task3Response,
                {
                    "relation": "strengthens",
                    "confidence": 1.0,
                    "evidence_phrase": "MAY export",
                    "brief_reason": "optional became mandatory",
                },
            ),
            (
                so.ExternalTask2Response,
                {
                    "external_item_id": "EXT0001",
                    "requirement": "The system SHOULD export reports.",
                    "modality": "recommended",
                    "confidence": 0.8,
                },
            ),
        ]

        for model, payload in cases:
            with self.subTest(model=model.__name__):
                parsed = model.model_validate(payload)
                self.assertEqual(parsed.model_dump(mode="json"), payload)

    def test_task_models_reject_invalid_confidence_values(self):
        valid_payloads = [
            (
                so.Task1Response,
                {"decision": "yes", "confidence": 0.9, "brief_reason": "explicit must"},
            ),
            (
                so.Task2Response,
                {
                    "requirement": "The system MAY export reports.",
                    "modality": "optional",
                    "confidence": 0.9,
                },
            ),
            (
                so.Task3Response,
                {
                    "relation": "preserves",
                    "confidence": 0.9,
                    "evidence_phrase": "MAY",
                    "brief_reason": "same modality",
                },
            ),
            (
                so.ExternalTask2Response,
                {
                    "external_item_id": "EXT0001",
                    "requirement": "The system MAY export reports.",
                    "modality": "optional",
                    "confidence": 0.9,
                },
            ),
        ]

        for model, payload in valid_payloads:
            for confidence in ["0.9", "90%", True, False, -0.01, 1.01, 95]:
                with self.subTest(model=model.__name__, confidence=confidence):
                    invalid = {**payload, "confidence": confidence}
                    with self.assertRaises(ValidationError):
                        model.model_validate(invalid)

    def test_task_models_reject_invalid_labels_missing_fields_and_extras(self):
        invalid_cases = [
            (so.Task1Response, {"decision": "maybe", "confidence": 0.9, "brief_reason": "hedged"}),
            (
                so.Task2Response,
                {"requirement": "The system SHOULD export reports.", "modality": "should", "confidence": 0.9},
            ),
            (
                so.Task3Response,
                {
                    "relation": "stronger",
                    "confidence": 0.9,
                    "evidence_phrase": "SHOULD",
                    "brief_reason": "wrong enum",
                },
            ),
            (so.Task1Response, {"confidence": 0.9, "brief_reason": "missing decision"}),
            (so.Task2Response, {"modality": "optional", "confidence": 0.9}),
            (so.Task3Response, {"relation": "preserves", "confidence": 0.9}),
            (
                so.ExternalTask2Response,
                {"requirement": "The system MAY export reports.", "modality": "optional", "confidence": 0.9},
            ),
            (
                so.Task2Response,
                {
                    "requirement": "The system MAY export reports.",
                    "modality": "optional",
                    "confidence": 0.9,
                    "unexpected": "field",
                },
            ),
        ]

        for model, payload in invalid_cases:
            with (
                self.subTest(model=model.__name__, payload=payload),
                self.assertRaises(ValidationError),
            ):
                model.model_validate(payload)

    def test_batch_models_validate_request_index_and_items(self):
        payload = {
            "results": [
                {
                    "request_index": 7,
                    "decision": "no",
                    "confidence": 0.85,
                    "brief_reason": "not mandatory",
                }
            ]
        }

        parsed = so.validated_payload_for_task("task1", payload, batched=True)

        self.assertEqual(parsed["results"][0]["request_index"], 7)
        self.assertEqual(parsed["results"][0]["confidence"], 0.85)

    def test_batch_models_reject_bad_items_and_extra_envelope_fields(self):
        invalid_batch = {
            "results": [
                {
                    "request_index": 7,
                    "decision": "yes",
                    "confidence": 95,
                    "brief_reason": "bad scale",
                }
            ]
        }
        extra_envelope = {
            "results": [
                {
                    "request_index": 7,
                    "requirement": "The system MAY export reports.",
                    "modality": "optional",
                    "confidence": 0.9,
                }
            ],
            "unexpected": "field",
        }
        extra_item = {
            "results": [
                {
                    "request_index": 7,
                    "relation": "preserves",
                    "confidence": 0.9,
                    "evidence_phrase": "MAY",
                    "brief_reason": "same modality",
                    "unexpected": "field",
                }
            ]
        }

        for task, payload in [
            ("task1", invalid_batch),
            ("task2", extra_envelope),
            ("task3", extra_item),
        ]:
            with (
                self.subTest(task=task),
                self.assertRaises(ValidationError),
            ):
                so.validated_payload_for_task(task, payload, batched=True)

    def test_validated_json_and_model_lookup_are_task_specific(self):
        text = json.dumps(
            {
                "requirement": "The system MAY export reports.",
                "modality": "optional",
                "confidence": 0.75,
            }
        )

        parsed = so.validated_json_for_task("task2", text)

        self.assertEqual(parsed["modality"], "optional")
        self.assertIs(so.response_model_for_task("task3"), so.Task3Response)
        self.assertIs(so.response_model_for_task("task2", batched=True), so.Task2BatchResponse)
        with self.assertRaisesRegex(ValueError, "Unsupported task"):
            so.response_model_for_task("task4")


if __name__ == "__main__":
    unittest.main()
