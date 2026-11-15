"""Representation metadata shared by training and offline inference."""

MEL = 'mel'
COMPLEX_STFT = 'complex_stft'


def representation(config):
    value = config['audio'].get('representation', MEL)
    if value not in (MEL, COMPLEX_STFT):
        raise ValueError(f'Unknown audio representation: {value}')
    return value


def feature_dim(config):
    audio = config['audio']
    return 2 * (audio['n_fft'] // 2 + 1) if representation(config) == COMPLEX_STFT else audio['n_mels']


def frame_count(config, samples):
    hop = config['audio']['hop_length']
    return samples // hop + 1 if representation(config) == COMPLEX_STFT else (samples + hop - 1) // hop


def feature_error_name(config):
    return 'generated_stft_ri_l1' if representation(config) == COMPLEX_STFT else 'generated_mel_l1'
