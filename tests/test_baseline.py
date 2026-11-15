import math
from pathlib import Path
import tempfile
import unittest

import soundfile as sf
import torch

from flowtse_rl.audio import BigVGANMel
from flowtse_rl.data import PrecomputedLibriMixDataset, FrameBatchSampler
from flowtse_rl.flow import flow_matching_loss, sample_ode, masked_mse
from flowtse_rl.model import MelUDiT
from flowtse_rl.vendor.bigvgan.meldataset import mel_spectrogram
from flowtse_rl.vocoder import FrozenBigVGAN


def tiny_model(conditioning_mode="channel_concat", attention_mode="bidirectional", use_checkpoint=False,
               skip_redundant_mask=True, vectorized_packing=True):
    torch.manual_seed(7)
    model = MelUDiT(mel_dim=4, hidden_size=32, depth=2, num_heads=4,
                    mlp_ratio=2, use_checkpoint=use_checkpoint,
                    conditioning_mode=conditioning_mode, attention_mode=attention_mode,
                    skip_redundant_mask=skip_redundant_mask, vectorized_packing=vectorized_packing)
    # Exercise real attention and FFN paths; zero initialization would make
    # padding invariance pass trivially for every input.
    with torch.no_grad():
        model.final_layer.linear.weight.normal_(std=0.03)
        for block in list(model.in_blocks) + [model.mid_block] + list(model.out_blocks):
            block.adaLN_modulation[-1].weight.normal_(std=0.02)
    return model


class FrontendTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.frontend = BigVGANMel()

    def test_matches_official_for_hop_aligned_audio(self):
        wave = torch.randn(2, 4096) * 0.1
        actual = self.frontend.analyze(wave)
        expected = mel_spectrogram(wave, 1024, 100, 24000, 256, 1024, 0)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    def test_ragged_analysis_matches_individual_and_covers_eos(self):
        lengths = torch.tensor([1, 431, 4097])
        wave = torch.randn(3, 4097) * 0.1
        for index, length in enumerate(lengths):
            wave[index, length:] = float('nan')
        actual, frames = self.frontend(wave, lengths)
        self.assertEqual(frames.tolist(), [math.ceil(int(n) / 256) for n in lengths])
        self.assertTrue(torch.isfinite(actual).all())
        for index, length in enumerate(lengths):
            expected = self.frontend.analyze(wave[index:index + 1, :length])
            torch.testing.assert_close(actual[index:index + 1, :, :frames[index]], expected)
            self.assertEqual(actual[index, :, frames[index]:].count_nonzero(), 0)


class ModelAndFlowTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_ragged_model_matches_single_and_ignores_nan_padding(self):
        model = tiny_model().eval()
        state, mixture, enrollment = torch.randn(2, 4, 9), torch.randn(2, 4, 9), torch.randn(2, 4, 5)
        lengths, elengths, time = torch.tensor([9, 6]), torch.tensor([5, 2]), torch.tensor([0.3, 0.7])
        actual = model(state, time, enrollment, mixture, lengths, elengths)
        for index in range(2):
            ml, el = int(lengths[index]), int(elengths[index])
            expected = model(state[index:index+1, :, :ml], time[index:index+1], enrollment[index:index+1, :, :el],
                             mixture[index:index+1, :, :ml], lengths[index:index+1], elengths[index:index+1])
            torch.testing.assert_close(actual[index:index+1, :, :ml], expected, atol=1e-6, rtol=1e-5)
        state[1, :, 6:] = float('nan')
        mixture[1, :, 6:] = float('nan')
        enrollment[1, :, 2:] = float('nan')
        torch.testing.assert_close(model(state, time, enrollment, mixture, lengths, elengths), actual)

    def test_flow_backward_ignores_padding(self):
        model = tiny_model().train()
        target, mixture, enrollment = torch.randn(2, 4, 9), torch.randn(2, 4, 9), torch.randn(2, 4, 5)
        loss = flow_matching_loss(model, target, mixture, enrollment, torch.tensor([9, 6]), torch.tensor([5, 2]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        error = torch.randn(2, 4, 9, requires_grad=True)
        masked_mse(error, torch.tensor([9, 6])).backward()
        self.assertEqual(error.grad[1, :, 6:].count_nonzero(), 0)

    def test_ode_reproducible_and_fixed_mixture(self):
        model = tiny_model().eval()
        mixture, enrollment = torch.randn(2, 4, 9), torch.randn(2, 4, 5)
        lengths, elengths = torch.tensor([9, 6]), torch.tensor([5, 2])
        original = mixture.clone()
        first = sample_ode(model, mixture, enrollment, lengths, elengths, steps=3, generator=torch.Generator().manual_seed(23))
        second = sample_ode(model, mixture, enrollment, lengths, elengths, steps=3, generator=torch.Generator().manual_seed(23))
        torch.testing.assert_close(first, second)
        torch.testing.assert_close(mixture, original)
        self.assertEqual(first[1, :, 6:].count_nonzero(), 0)

    def test_vocoder_ignores_padded_context_at_real_tail(self):
        class ContextDecoder(torch.nn.Module):
            def forward(self, features):
                filtered = torch.nn.functional.conv1d(features[:, :1], features.new_ones(1, 1, 3), padding=1)
                return filtered.repeat_interleave(256, dim=-1)
        vocoder = FrozenBigVGAN.__new__(FrozenBigVGAN)
        vocoder.model = ContextDecoder()
        mel = torch.ones(2, 100, 4)
        mel[0, :, 2:] = 999
        actual = vocoder(mel, torch.tensor([511, 1023]))
        expected = vocoder.model(mel[:1, :, :2])[0, 0, :511]
        torch.testing.assert_close(actual[0], expected)
        self.assertEqual([len(wave) for wave in actual], [511, 1023])


class DatasetTests(unittest.TestCase):
    def test_short_enrollment_keeps_both_target_pairs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            split = root / 'dev'
            split.mkdir()
            metadata = root / 'metadata'
            metadata.mkdir()
            paths = []
            for name, samples in [('mix', 3200), ('one', 3200), ('two', 3200), ('enroll1', 800), ('enroll2', 1000)]:
                path = root / f'{name}.wav'
                sf.write(path, (torch.randn(samples) * 0.03).numpy(), 16000, subtype='FLOAT')
                paths.append(path)
            (metadata/'mixture_dev_mix_both.csv').write_text('mixture_ID,mixture_path,source_1_path,source_2_path,length\n'
                + f'1-2-3_4-5-6,{paths[0]},{paths[1]},{paths[2]},3200\n')
            (split/'mixture2enrollment.csv').write_text('mixture_id,utterance_id,enr_path1,length1\n'
                + f'1-2-3_4-5-6,1-2-3,{paths[3]},800\n1-2-3_4-5-6,4-5-6,{paths[4]},1000\n')
            config = dict(seed=42, input_sample_rate=16000, enrollment_seconds=[3, 3], segment_seconds=None)
            dataset = PrecomputedLibriMixDataset([split], metadata, {'sample_rate':24000,'target_rms':0.1}, config, False)
            self.assertEqual(len(dataset), 2)
            self.assertEqual(dataset[0]['enrollment'].numel(), 1200)
            self.assertEqual(dataset[1]['enrollment'].numel(), 1500)
            self.assertEqual(dataset[0]['target'].numel(), dataset[0]['mixture'].numel())


if __name__ == '__main__':
    unittest.main()

