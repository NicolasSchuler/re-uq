import json
import unittest

from pydantic import Field, ValidationError

from scripts import eval_utils as eu, structured_outputs as so

# The exact bytes of the six provider JSON Schemas as they were sent by every
# archived run, captured before the schemas were derived from the Pydantic
# models. They are part of the request payload and therefore of the config
# fingerprint that `--mode resume` matches cached rows on: a single reordered
# key here would re-request every archived structured-output row. Treat a
# failure of the equality test below as "the derivation drifted", never as
# "the pin is stale", unless the contract is being changed on purpose.
PINNED_PROVIDER_SCHEMAS = {
    ("task1", False): (
        '{"type":"object","properties":{"decision":{"type":"string","enum":["'
        'yes","no"]},"confidence":{"type":"number","minimum":0,"maximum":1},"'
        'brief_reason":{"type":"string","maxLength":200}},"required":["decisi'
        'on","confidence","brief_reason"],"additionalProperties":false}'
    ),
    ("task1", True): (
        '{"type":"object","properties":{"results":{"type":"array","items":{"t'
        'ype":"object","properties":{"request_index":{"type":"integer"},"deci'
        'sion":{"type":"string","enum":["yes","no"]},"confidence":{"type":"nu'
        'mber","minimum":0,"maximum":1},"brief_reason":{"type":"string","maxL'
        'ength":200}},"required":["request_index","decision","confidence","br'
        'ief_reason"],"additionalProperties":false}}},"required":["results"],'
        '"additionalProperties":false}'
    ),
    ("task2", False): (
        '{"type":"object","properties":{"requirement":{"type":"string"},"moda'
        'lity":{"type":"string","enum":["mandatory","recommended","optional",'
        '"nice_to_have"]},"confidence":{"type":"number","minimum":0,"maximum"'
        ':1}},"required":["requirement","modality","confidence"],"additionalP'
        'roperties":false}'
    ),
    ("task2", True): (
        '{"type":"object","properties":{"results":{"type":"array","items":{"t'
        'ype":"object","properties":{"request_index":{"type":"integer"},"requ'
        'irement":{"type":"string"},"modality":{"type":"string","enum":["mand'
        'atory","recommended","optional","nice_to_have"]},"confidence":{"type'
        '":"number","minimum":0,"maximum":1}},"required":["request_index","re'
        'quirement","modality","confidence"],"additionalProperties":false}}},'
        '"required":["results"],"additionalProperties":false}'
    ),
    ("task3", False): (
        '{"type":"object","properties":{"relation":{"type":"string","enum":["'
        'preserves","strengthens","weakens","content_changed"]},"confidence":'
        '{"type":"number","minimum":0,"maximum":1},"evidence_phrase":{"type":'
        '"string","maxLength":240},"brief_reason":{"type":"string","maxLength'
        '":240}},"required":["relation","confidence","evidence_phrase","brief'
        '_reason"],"additionalProperties":false}'
    ),
    ("task3", True): (
        '{"type":"object","properties":{"results":{"type":"array","items":{"t'
        'ype":"object","properties":{"request_index":{"type":"integer"},"rela'
        'tion":{"type":"string","enum":["preserves","strengthens","weakens","'
        'content_changed"]},"confidence":{"type":"number","minimum":0,"maximu'
        'm":1},"evidence_phrase":{"type":"string","maxLength":240},"brief_rea'
        'son":{"type":"string","maxLength":240}},"required":["request_index",'
        '"relation","confidence","evidence_phrase","brief_reason"],"additiona'
        'lProperties":false}}},"required":["results"],"additionalProperties":'
        "false}"
    ),
}


def _schema_bytes(schema):
    return json.dumps(schema, ensure_ascii=True, separators=(",", ":"))


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
            (
                so.Task1Response,
                {"decision": "maybe", "confidence": 0.9, "brief_reason": "hedged"},
            ),
            (
                so.Task2Response,
                {
                    "requirement": "The system SHOULD export reports.",
                    "modality": "should",
                    "confidence": 0.9,
                },
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
                {
                    "requirement": "The system MAY export reports.",
                    "modality": "optional",
                    "confidence": 0.9,
                },
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
        self.assertIs(
            so.response_model_for_task("task2", batched=True), so.Task2BatchResponse
        )
        with self.assertRaisesRegex(ValueError, "Unsupported task"):
            so.response_model_for_task("task4")


class ResponseContractDriftTest(unittest.TestCase):
    """The Pydantic models and the provider JSON Schema must agree."""

    def test_task3_brief_reason_is_required_like_the_schema_says(self):
        payload = {
            "relation": "preserves",
            "confidence": 0.9,
            "evidence_phrase": "MAY export",
        }

        for model, missing in (
            (so.Task3Response, payload),
            (so.Task3BatchItem, {**payload, "request_index": 0}),
        ):
            with self.subTest(model=model.__name__), self.assertRaises(ValidationError):
                model.model_validate(missing)

        self.assertIn("brief_reason", so.provider_schema_for_task("task3")["required"])
        self.assertNotIn(
            "default",
            so.Task3Response.model_json_schema()["properties"]["brief_reason"],
        )

    def test_task2_requirement_rejects_blank_and_whitespace_only_strings(self):
        for requirement in ["", "   ", "\n\t "]:
            for model in (so.Task2Response, so.ExternalTask2Response):
                payload = {
                    "requirement": requirement,
                    "modality": "optional",
                    "confidence": 0.9,
                }
                if model is so.ExternalTask2Response:
                    payload["external_item_id"] = "EXT0001"
                with (
                    self.subTest(model=model.__name__, requirement=requirement),
                    self.assertRaises(ValidationError),
                ):
                    model.model_validate(payload)

        with self.assertRaises(ValidationError):
            so.validated_payload_for_task(
                "task2",
                {
                    "results": [
                        {
                            "request_index": 0,
                            "requirement": " ",
                            "modality": "optional",
                            "confidence": 0.9,
                        }
                    ]
                },
                batched=True,
            )

    def test_blank_requirement_rejection_stays_out_of_the_provider_schema(self):
        # `Field(min_length=1)` would express the same rule but would add
        # "minLength" to the schema and move bytes we have already sent.
        requirement = so.provider_schema_for_task("task2")["properties"]["requirement"]

        self.assertEqual(requirement, {"type": "string"})


class ProviderSchemaDerivationTest(unittest.TestCase):
    def test_derived_schemas_are_byte_identical_to_the_pinned_contract(self):
        for (task, batched), pinned in PINNED_PROVIDER_SCHEMAS.items():
            with self.subTest(task=task, batched=batched):
                self.assertEqual(
                    _schema_bytes(so.provider_schema_for_task(task, batched=batched)),
                    pinned,
                )
                # The request builder must reach the same bytes, since that is
                # what actually goes into `response_format`.
                self.assertEqual(
                    _schema_bytes(eu.task_response_schema(task, batched=batched)),
                    pinned,
                )

    def test_response_format_carries_the_derived_schema(self):
        response_format = eu.response_format_for_task(
            "task3", "json_schema", batched=True
        )

        self.assertEqual(response_format["json_schema"]["name"], "re_uq_task3_batch")
        self.assertTrue(response_format["json_schema"]["strict"])
        self.assertEqual(
            _schema_bytes(response_format["json_schema"]["schema"]),
            PINNED_PROVIDER_SCHEMAS[("task3", True)],
        )

    def test_derivation_inlines_refs_and_drops_pydantic_bookkeeping(self):
        for task in ("task1", "task2", "task3"):
            for batched in (False, True):
                with self.subTest(task=task, batched=batched):
                    text = _schema_bytes(
                        so.provider_schema_for_task(task, batched=batched)
                    )
                    for absent in ('"$defs"', '"$ref"', '"title"', '"default"'):
                        self.assertNotIn(absent, text)

    def test_derivation_refuses_constraints_it_cannot_place(self):
        class PatternedResponse(so.StrictResponseModel):
            label: str = Field(pattern="^a")

        with self.assertRaisesRegex(ValueError, "No provider-contract position"):
            so.provider_json_schema(PatternedResponse)

    def test_label_sets_are_shared_with_the_analysis_constants(self):
        self.assertEqual(
            so.provider_schema_for_task("task2")["properties"]["modality"]["enum"],
            list(eu.MODALITIES),
        )
        self.assertEqual(
            so.provider_schema_for_task("task3")["properties"]["relation"]["enum"],
            list(eu.TASK3_RELATIONS),
        )

    def test_unsupported_task_still_reports_the_schema_specific_error(self):
        with self.assertRaisesRegex(ValueError, "Unsupported task for JSON Schema"):
            eu.task_response_schema("task4")


class ParseRepairRecordingTest(unittest.TestCase):
    """A conforming response has to be distinguishable from a repaired one."""

    def _parse(self, task, payload_text):
        return eu.parse_task_response(task, payload_text, eu.CONFIDENCE_SCALE_0_1)

    def test_conforming_responses_record_no_repairs(self):
        cases = {
            "task1": {"decision": "yes", "confidence": 0.9, "brief_reason": "must"},
            "task2": {
                "requirement": "The system MAY export reports.",
                "modality": "optional",
                "confidence": 0.9,
            },
            "task3": {
                "relation": "preserves",
                "confidence": 0.9,
                "evidence_phrase": "MAY export",
                "brief_reason": "same modality",
            },
        }

        for task, payload in cases.items():
            with self.subTest(task=task):
                parsed, status = self._parse(task, json.dumps(payload))
                self.assertEqual(status, "ok")
                self.assertEqual(parsed[eu.PARSE_REPAIRS_FIELD], [])

    def test_batched_items_keep_their_request_index_unflagged(self):
        payload = {
            "request_index": 3,
            "requirement": "The system MAY export reports.",
            "modality": "optional",
            "confidence": 0.9,
        }

        parsed, status = self._parse("task2", json.dumps(payload))

        self.assertEqual(status, "ok")
        self.assertEqual(parsed[eu.PARSE_REPAIRS_FIELD], [])

    def test_each_tolerated_deviation_is_named(self):
        parsed, status = self._parse(
            "task3",
            'Here you go:\n{"relation":"weakened","confidence":0.9,'
            f'"evidence_phrase":"{"MAY " * 80}","source":"extra"}}\nHope that helps.',
        )

        self.assertEqual(status, "ok")
        self.assertEqual(parsed["relation"], "weakens")
        self.assertEqual(len(parsed["evidence_phrase"]), 240)
        self.assertEqual(parsed["brief_reason"], "")
        self.assertEqual(
            parsed[eu.PARSE_REPAIRS_FIELD],
            [
                eu.PARSE_REPAIR_PROSE_WRAPPER,
                eu.PARSE_REPAIR_UNEXPECTED_FIELDS,
                eu.PARSE_REPAIR_LABEL_ALIAS,
                "truncated_evidence_phrase",
                "defaulted_brief_reason",
            ],
        )

    def test_non_string_free_text_is_coerced_and_recorded(self):
        parsed, status = self._parse(
            "task1", '{"decision":"yes","confidence":0.9,"brief_reason":7}'
        )

        self.assertEqual(status, "ok")
        self.assertEqual(parsed["brief_reason"], "7")
        self.assertEqual(parsed[eu.PARSE_REPAIRS_FIELD], ["coerced_brief_reason"])

    def test_blank_task2_requirement_is_a_parse_failure(self):
        for requirement in ["", "   "]:
            with self.subTest(requirement=requirement):
                _, status = self._parse(
                    "task2",
                    json.dumps(
                        {
                            "requirement": requirement,
                            "modality": "optional",
                            "confidence": 0.9,
                        }
                    ),
                )
                self.assertEqual(status, "missing_fields")

    def test_failed_parses_still_carry_the_repairs_seen_so_far(self):
        parsed, status = self._parse(
            "task2", 'note {"modality":"nope","confidence":0.9,"requirement":"x"}'
        )

        self.assertEqual(status, "invalid_label")
        self.assertEqual(
            parsed[eu.PARSE_REPAIRS_FIELD], [eu.PARSE_REPAIR_PROSE_WRAPPER]
        )

    def test_instructor_path_reports_no_repairs_unless_prose_was_stripped(self):
        payload = json.dumps(
            {
                "requirement": "The system MAY export reports.",
                "modality": "optional",
                "confidence": 0.9,
            }
        )

        parsed, status = eu.parse_instructor_task_response("task2", payload)
        self.assertEqual(status, "ok")
        self.assertEqual(parsed[eu.PARSE_REPAIRS_FIELD], [])

        wrapped, status = eu.parse_instructor_task_response(
            "task2", f"```json\n{payload}\n```"
        )
        self.assertEqual(status, "ok")
        self.assertEqual(
            wrapped[eu.PARSE_REPAIRS_FIELD], [eu.PARSE_REPAIR_PROSE_WRAPPER]
        )

    def test_repairs_are_readable_back_off_raw_rows(self):
        rows = [
            {"parsed_json": {eu.PARSE_REPAIRS_FIELD: []}},
            {"parsed_json": {eu.PARSE_REPAIRS_FIELD: ["label_alias"]}},
            {"parsed_json": {eu.PARSE_REPAIRS_FIELD: ["label_alias", "prose_wrapper"]}},
            # Rows written before the field existed, and failed parses.
            {"parsed_json": {"relation": "preserves"}},
            {"parsed_json": None},
        ]

        self.assertEqual(
            eu.parse_repairs_for_record(rows[2]), ["label_alias", "prose_wrapper"]
        )
        self.assertEqual(eu.parse_repairs_for_record(rows[3]), [])
        self.assertEqual(eu.parse_repairs_for_record(rows[4]), [])
        self.assertEqual(
            eu.parse_repair_counts(rows),
            {"repaired_records": 2, "label_alias": 2, "prose_wrapper": 1},
        )
        self.assertEqual(eu.parse_repair_counts([]), {"repaired_records": 0})


if __name__ == "__main__":
    unittest.main()
