"""The detokenizer keeps each request's decode state from the DetokenizeHeader
on its first output and decodes every later row from that row's output_ids."""

import unittest
from array import array

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.environ import envs
from sglang.srt.managers.detokenizer_manager import (
    DetokenizerManager,
    LimitedCapacityDict,
)
from sglang.srt.managers.io_struct import (
    BatchTokenIDOutput,
    BeamSearchOutput,
    DetokenizeHeader,
    DetokenizeReleaseReq,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

SPECIAL = 100


class _Tokenizer:
    """Renders each id; ids from SPECIAL up are special tokens."""

    is_fast = True

    def decode(self, ids, skip_special_tokens, spaces_between_special_tokens):
        return "".join(
            f"<{t}>" if t >= SPECIAL else f"{t}."
            for t in ids
            if not (skip_special_tokens and t >= SPECIAL)
        )

    def batch_decode(self, ids_list, **kwargs):
        return [self.decode(ids, **kwargs) for ids in ids_list]


def _header(prompt_tail_ids, *, skip_special_tokens=True, ipc="worker-a"):
    return DetokenizeHeader(
        prompt_tail_ids=array("q", prompt_tail_ids),
        skip_special_tokens=skip_special_tokens,
        spaces_between_special_tokens=True,
        no_stop_trim=False,
        http_worker_ipc=ipc,
    )


def _output(sent, rows):
    """rows: (rid, output_ids, header or None, finished). sent: by rid, the
    output ids the scheduler sent so far, which sets each row's
    completion_tokens."""
    n = len(rows)
    completion_tokens = []
    for rid, ids, header, _ in rows:
        sent[rid] = (0 if header is not None else sent.get(rid, 0)) + len(ids)
        completion_tokens.append(sent[rid])
    headers = [header for _, _, header, _ in rows]
    return BatchTokenIDOutput(
        rids=[rid for rid, _, _, _ in rows],
        finished_reasons=[
            {"type": "length"} if finished else None for *_, finished in rows
        ],
        output_ids=[array("q", ids) for _, ids, _, _ in rows],
        detokenize_headers=headers if any(headers) else None,
        reasoning_tokens=[0] * n,
        completion_tokens=completion_tokens,
        input_token_logprobs_val=None,
        input_token_logprobs_idx=None,
        output_token_logprobs_val=None,
        output_token_logprobs_idx=None,
        input_top_logprobs_val=None,
        input_top_logprobs_idx=None,
        output_top_logprobs_val=None,
        output_top_logprobs_idx=None,
        input_token_ids_logprobs_val=None,
        input_token_ids_logprobs_idx=None,
        output_token_ids_logprobs_val=None,
        output_token_ids_logprobs_idx=None,
        output_token_entropy_val=None,
        output_token_sampling_mask=None,
        output_hidden_states=None,
        routed_experts=None,
        indexer_topk=None,
        placeholder_tokens_idx=None,
        placeholder_tokens_val=None,
    )


def _detokenizer():
    detok = DetokenizerManager.__new__(DetokenizerManager)
    detok.tokenizer = _Tokenizer()
    detok.vocab_size = None
    detok.decode_status = LimitedCapacityDict(capacity=16)
    detok.disable_tokenizer_batch_decode = False
    detok.is_tool_call_parser_gpt_oss = False
    detok.tool_close_token_ids = frozenset()
    return detok


class TestDetokenizerDecodeStatus(CustomTestCase):
    def test_later_rows_decode_with_the_first_rows_header(self):
        """Rows after the first carry no header: their text continues from the
        stored prompt tail with the stored flags, and their http_worker_ipcs
        come from the stored header."""
        detok = _detokenizer()
        sent = {}
        header = _header([1, 2], skip_special_tokens=False)

        steps = [
            ([3], header, False),
            ([SPECIAL, 4], None, False),
            ([5], None, True),
        ]
        texts = []
        for ids, row_header, finished in steps:
            out = detok.handle_batch_token_id_out(
                _output(sent, [("r", ids, row_header, finished)])
            )
            self.assertEqual(out.http_worker_ipcs, ["worker-a"])
            texts += out.output_strs

        self.assertEqual(texts, ["3.", f"<{SPECIAL}>4.", "5."])
        self.assertNotIn("r", detok.decode_status)

    def test_new_header_replaces_the_state_of_a_reused_rid(self):
        """A request that reuses the rid of one that left state behind decodes
        only its own ids."""
        detok = _detokenizer()
        sent = {}
        detok.handle_batch_token_id_out(
            _output(sent, [("r", [3], _header([1]), False)])
        )

        out = detok.handle_batch_token_id_out(
            _output(sent, [("r", [7], _header([6], ipc="worker-b"), False)])
        )

        self.assertEqual(out.output_strs, ["7."])
        self.assertEqual(out.http_worker_ipcs, ["worker-b"])
        self.assertEqual(list(detok.decode_status["r"].decode_ids), [6, 7])

    def test_a_row_without_state_drops_only_its_text(self):
        """After a release, a late row of the request decodes to no text and a
        warning, while the other rows of the batch decode as usual."""
        detok = _detokenizer()
        sent = {}
        detok.handle_batch_token_id_out(
            _output(
                sent,
                [
                    ("gone", [3], _header([1]), False),
                    ("kept", [4], _header([2]), False),
                ],
            )
        )

        detok.handle_detokenize_release_req(DetokenizeReleaseReq(rid="gone"))
        self.assertNotIn("gone", detok.decode_status)
        with self.assertLogs(
            "sglang.srt.managers.detokenizer_manager", level="WARNING"
        ) as logs:
            out = detok.handle_batch_token_id_out(
                _output(sent, [("gone", [5], None, True), ("kept", [6], None, False)])
            )

        self.assertIn("rid='gone'", logs.output[0])
        self.assertEqual(out.output_strs, ["", "6."])
        self.assertEqual(out.http_worker_ipcs, [None, "worker-a"])

    def test_a_row_that_does_not_continue_the_received_ids_raises_an_alarm(self):
        """Each row's completion_tokens is the ids received before plus its own.
        A row that breaks it logs a warning (raises in CI) but is still decoded,
        and the count resyncs to it; a header restarts the count; a beam row,
        whose completion_tokens counts its sequences, is not checked."""
        detok = _detokenizer()
        sent = {}
        detok.handle_batch_token_id_out(
            _output(sent, [("r", [3], _header([1]), False)])
        )
        detok.handle_batch_token_id_out(
            _output(sent, [("r", [4], _header([2]), False)])
        )

        skipped = _output({"r": 5}, [("r", [5], None, False)])
        with envs.SGLANG_IS_IN_CI.override(True), self.assertRaises(RuntimeError):
            detok.handle_batch_token_id_out(skipped)
        with self.assertLogs(
            "sglang.srt.managers.detokenizer_manager", level="WARNING"
        ) as logs:
            out = detok.handle_batch_token_id_out(skipped)
        self.assertIn("do not continue the 1 received", logs.output[0])
        self.assertEqual(out.output_strs, ["5."])
        self.assertEqual(out.http_worker_ipcs, ["worker-a"])

        with self.assertNoLogs(
            "sglang.srt.managers.detokenizer_manager", level="WARNING"
        ):
            out = detok.handle_batch_token_id_out(
                _output({"r": 6}, [("r", [7], None, False)])
            )
        self.assertEqual(out.output_strs, ["7."])

        beam = _output(sent, [("r", [8], None, True)])
        beam.completion_tokens = [9]
        beam.beam_search_output = [BeamSearchOutput(sequences=[])]
        out = detok.handle_batch_token_id_out(beam)
        self.assertEqual(out.output_strs, ["8."])


if __name__ == "__main__":
    unittest.main()
