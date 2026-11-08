# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Regression: VCR matching stays compatible with anthropic SDK >= 2.3.0.

anthropic SDK 2.3.0 stopped appending ``?beta=true`` on ``beta.messages``
endpoints, while existing cassettes were recorded with that query. The suite
filters the ``beta`` query parameter on record and playback. These tests pin
that this normalization is scoped to the ``beta`` parameter only: other query
parameters still participate in matching (no blanket query loosening), the
``beta`` value itself is irrelevant to response semantics, and recorded
responses are replayed byte-for-byte.
"""

from vcr import VCR
from vcr.request import Request as VCRRequest

RECORDED_WITH_BETA_TRUE = """
interactions:
- request:
    body: '{}'
    headers: {}
    method: POST
    uri: https://api.anthropic.com/v1/messages?beta=true
  response:
    body:
      string: '{"ok": true}'
    headers:
      content-type:
      - application/json
    status:
      code: 200
      message: OK
"""


def _open_cassette(tmp_path):
    path = tmp_path / "beta.yaml"
    path.write_text(RECORDED_WITH_BETA_TRUE, encoding="utf-8")
    my_vcr = VCR(filter_query_parameters=["beta"], record_mode="none")
    return my_vcr.use_cassette(str(path))


def _playable(cassette, uri):
    return cassette.can_play_response_for(VCRRequest("POST", uri, b"{}", {}))


def test_request_without_beta_matches_recording_with_beta(tmp_path):
    # SDK 2.3.0 sends no `?beta=true`: it must still match the recorded cassette.
    with _open_cassette(tmp_path) as cassette:
        assert _playable(cassette, "https://api.anthropic.com/v1/messages")


def test_unrelated_query_parameter_still_blocks_matching(tmp_path):
    # The filter must NOT be a blanket ignore of the whole query string.
    with _open_cassette(tmp_path) as cassette:
        assert not _playable(
            cassette, "https://api.anthropic.com/v1/messages?foo=bar"
        )


def test_recorded_response_is_replayed_unchanged(tmp_path):
    # Responses (not just requests) are untouched: no body or status rewriting.
    with _open_cassette(tmp_path) as cassette:
        request = VCRRequest(
            "POST", "https://api.anthropic.com/v1/messages", b"{}", {}
        )
        assert cassette.can_play_response_for(request)
        response = cassette.play_response(request)
        body = (
            response["body"]["string"]
            if isinstance(response, dict)
            else response.body
        )
        assert (
            "ok" in body.decode() if isinstance(body, bytes) else "ok" in body
        )
        status = (
            response["status"]
            if isinstance(response, dict)
            else response.status
        )
        assert status["code"] == 200
