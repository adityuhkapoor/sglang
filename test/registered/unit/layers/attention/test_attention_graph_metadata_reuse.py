"""CUDA graph variants must retain the metadata storage replay refreshes."""

import unittest
from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.attention import flashinfer_mla_backend as mla
from sglang.srt.layers.attention import trtllm_mla_backend as trtllm_mla
from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


class TestFlashInferCudaGraphMetadataReuse(unittest.TestCase):
    def test_variants_share_wrappers_within_each_graph_family(self):
        for mode in (
            ForwardMode.DECODE,
            ForwardMode.TARGET_VERIFY,
            ForwardMode.DLLM_EXTEND,
            ForwardMode.DRAFT_EXTEND_V2,
        ):
            with self.subTest(mode=mode):
                backend = FlashInferAttnBackend.__new__(FlashInferAttnBackend)
                backend.decode_cuda_graph_metadata = {}
                backend.prefill_cuda_graph_metadata = {}
                backend.draft_extend_cuda_graph_metadata = {}
                backend._create_decode_wrappers = lambda bs, num_tokens: [object()]
                backend._create_prefill_wrappers = lambda bs, use_custom_mask=False: [
                    object()
                ]
                field = "decode_wrappers" if mode.is_decode() else "prefill_wrappers"
                backend._prepare_cuda_graph_metadata(2, 2, mode, None)
                first = getattr(backend.forward_metadata, field)
                backend._prepare_cuda_graph_metadata(2, 2, mode, None)
                self.assertIs(getattr(backend.forward_metadata, field), first)
                backend._prepare_cuda_graph_metadata(4, 4, mode, None)
                self.assertIsNot(getattr(backend.forward_metadata, field), first)
                backend._prepare_cuda_graph_metadata(2, 2, mode, None)
                self.assertIs(getattr(backend.forward_metadata, field), first)

    def test_mla_replay_refreshes_the_wrappers_of_both_captured_variants(self):
        for mode in (ForwardMode.DECODE, ForwardMode.TARGET_VERIFY):
            with self.subTest(mode=mode):
                backend = mla.FlashInferMLAAttnBackend.__new__(
                    mla.FlashInferMLAAttnBackend
                )
                backend.decode_cuda_graph_metadata = {}
                backend.prefill_cuda_graph_metadata = {}
                backend.workspace_buffer = object()
                backend.cuda_graph_qo_indptr = torch.arange(3, dtype=torch.int32)
                backend.cuda_graph_kv_indptr = torch.zeros(3, dtype=torch.int32)
                backend.cuda_graph_kv_indices = torch.zeros(32, dtype=torch.int32)
                backend.cuda_graph_kv_lens = torch.zeros(2, dtype=torch.int32)
                backend.cuda_graph_qo_indptr_cpu = backend.cuda_graph_qo_indptr.clone()
                backend.cuda_graph_kv_indptr_cpu = backend.cuda_graph_kv_indptr.clone()
                backend.fast_decode_kwargs = {}
                backend.fast_plan_qo_indptr_cpu = torch.zeros(3, dtype=torch.int32)
                backend.fast_plan_kv_indptr_cpu = torch.zeros(3, dtype=torch.int32)
                backend.fast_plan_kv_len_arr_cpu = torch.zeros(2, dtype=torch.int32)

                def update_decode(lengths, *args, decode_wrapper, **kwargs):
                    if not kwargs["init_metadata_replay"]:
                        self.assertNotIsInstance(decode_wrapper.plan, partial)
                    decode_wrapper.lengths.copy_(lengths)

                def update_prefill(
                    indices, lengths, *args, prefill_wrapper_paged, **kwargs
                ):
                    prefill_wrapper_paged.lengths.copy_(lengths)

                backend.indices_updater_decode = SimpleNamespace(update=update_decode)
                backend.indices_updater_prefill = SimpleNamespace(update=update_prefill)
                lengths = torch.tensor([1, 1], dtype=torch.int64)
                batch = SimpleNamespace(
                    batch_size=2,
                    req_pool_indices=torch.arange(2),
                    seq_lens=lengths,
                    seq_lens_cpu=lengths,
                    seq_lens_sum=2,
                    positions=torch.arange(2),
                    forward_mode=mode,
                    spec_info=SimpleNamespace(
                        draft_token_num=1,
                        spec_input_type=mla.SpecInputType.EAGLE_VERIFY,
                    ),
                    ragged_verify_layout=None,
                    iteration_ids=None,
                )
                with (
                    patch.object(
                        mla,
                        "BatchMLAPagedAttentionWrapper",
                        side_effect=lambda *a, **k: SimpleNamespace(
                            lengths=torch.zeros(2, dtype=torch.int64), plan=Mock()
                        ),
                        create=True,
                    ),
                    patch.object(mla, "update_local_kv_lens_for_dcp"),
                ):
                    backend.init_forward_metadata_out_graph(batch, in_capture=True)
                    field = "decode_wrapper" if mode.is_decode() else "prefill_wrapper"
                    first = getattr(backend.forward_metadata, field)
                    backend.init_forward_metadata_out_graph(batch, in_capture=True)
                    lengths.copy_(torch.tensor([7, 11]))
                    batch.seq_lens_sum = 18
                    backend.init_forward_metadata_out_graph(batch)

                    torch.testing.assert_close(first.lengths, lengths)


class TestTRTLLMMLAGraphMetadata(unittest.TestCase):
    @patch.object(
        trtllm_mla, "get_parallel", return_value=SimpleNamespace(dcp_enabled=False)
    )
    def test_replay_updates_captured_storage_and_isolates_keys(self, _parallel):
        backend = trtllm_mla.TRTLLMMLABackend.__new__(trtllm_mla.TRTLLMMLABackend)
        backend.decode_cuda_graph_metadata = {}
        backend.decode_cuda_graph_kv_indices = torch.zeros((4, 4), dtype=torch.int32)
        backend.max_context_len = 32
        backend.num_draft_tokens = 3
        backend._calc_padded_blocks = lambda _: 4
        backend.runner_ids = SimpleNamespace(
            is_translating=True,
            copy_page_table=lambda plan, out: out.copy_(plan[: out.shape[0]]),
        )
        captured = []
        for index, mode in enumerate(
            (
                ForwardMode.DECODE,
                ForwardMode.TARGET_VERIFY,
                ForwardMode.DRAFT_EXTEND_V2,
            )
        ):
            with self.subTest(mode=mode):
                lengths = torch.ones(2, dtype=torch.int64)
                num_tokens = 2 if mode.is_decode_or_idle() else 6
                backend._init_cuda_graph_metadata(
                    2, num_tokens, mode, lengths, lengths.device
                )
                first = backend.forward_decode_metadata
                captured.append(first.seq_lens_k)
                backend._init_cuda_graph_metadata(
                    2, num_tokens, mode, lengths, lengths.device
                )
                page_table = (
                    torch.arange(8, dtype=torch.int32).reshape(2, 4) + 8 * index
                )
                lengths.copy_(torch.tensor([7 + index, 11 + index]))
                backend._apply_cuda_graph_metadata(
                    page_table, 2, torch.tensor([0, 1]), lengths, mode
                )

                expected = lengths.to(torch.int32)
                if mode.is_target_verify():
                    expected += backend.num_draft_tokens
                torch.testing.assert_close(first.seq_lens_k, expected)
                torch.testing.assert_close(first.block_kv_indices, page_table)

        lengths = torch.ones(4, dtype=torch.int64)
        backend._init_cuda_graph_metadata(
            4, 4, ForwardMode.DECODE, lengths, lengths.device
        )
        captured.append(backend.forward_decode_metadata.seq_lens_k)
        self.assertEqual(len({buffer.data_ptr() for buffer in captured}), 4)
        backend._init_cuda_graph_metadata(
            2, 2, ForwardMode.IDLE, lengths[:2], lengths.device
        )
        self.assertEqual(
            backend.forward_decode_metadata.seq_lens_k.data_ptr(),
            captured[0].data_ptr(),
        )


if __name__ == "__main__":
    unittest.main()
