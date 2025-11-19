from ast import mod
from wave import WAVE_FORMAT_PCM
import numpy as np
import scipy.io.wavfile as wav
import librosa
import librosa.display
import matplotlib.pyplot as plt
import torch
import os
from itertools import combinations
from datasets import Dataset, Audio
from utils.util import read_json,to_pickle,to_json
from transformers import LlamaTokenizer,AutoTokenizer,AutoProcessor,EncodecModel

encodec_model = EncodecModel.from_pretrained("facebook/encodec_24khz")
processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")
access_token = ''
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B",token=access_token)


# config = "./configs/repcodec_dim768.yaml"
# repcodec_trained_model_path = './configs/checkpoint-100000steps.pkl'


def frequency_dom(wav_file):
    fs, signal = wav.read(wav_file) 
    wav_name = os.path.basename(wav_file)

    if signal.ndim > 1:
        signal = signal[:, 0]  
    N = len(signal) 
    freqs = np.fft.fftfreq(N, 1/fs) 
    fft_values = np.fft.fft(signal) 

    positive_freqs = freqs[:N//2]
    magnitude = np.abs(fft_values[:N//2])  

    plt.figure(figsize=(8, 4))

    plt.plot(positive_freqs, magnitude)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('Frequency Domain Representation')
    plt.grid()
    plt.savefig('./spec/{}_domain.png'.format(wav_name))

def frequency_dom_former(wav_file):
    fs, signal = wav.read(wav_file) 
    wav_name = os.path.basename(wav_file)

    if signal.ndim > 1:
        signal = signal[:, 0]  

    N = len(signal)
    fft_values = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(N, d=1/fs)

    mask = (frequencies >= 0) & (frequencies <= 500)
    plt.figure(figsize=(8, 4))

    plt.plot(frequencies[mask], np.abs(fft_values[mask]))

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('Frequency Domain Representation')
    plt.grid()
    plt.savefig('./spec/former/{}_domain.png'.format(wav_name))

def encodec_token(wav_file):

    # waveform, sr = torchaudio.load(wav_file)

    data_dict = {"audio": [wav_file]}
    data_dataset = Dataset.from_dict(data_dict).cast_column("audio", Audio())

    # breakpoint()
    audio_sample = data_dataset[-1]["audio"]["array"]
    inputs = processor(raw_audio=audio_sample, sampling_rate=24000, return_tensors="pt")

    with torch.no_grad():
        encoded_frames = encodec_model.encode(inputs["input_values"], inputs["padding_mask"])

    print(encoded_frames.audio_codes.shape)
    tokens = encoded_frames.audio_codes[0][0]
    # tokens = [frame[0] for frame in encoded_frames]
    tokens_list = [str(token) for token in tokens[0].tolist()]

    return tokens_list


def fixed_spec_binning(wav_file):

    wav_name = os.path.basename(wav_file)
    y, sr = librosa.load(wav_file, sr=None) 

    D = librosa.stft(y, n_fft=1024, hop_length=512)  
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max) 

    num_bins = 128  
    freq_bins = np.linspace(np.min(S_db), np.max(S_db), num_bins)  

    quantized_tokens = np.digitize(S_db, bins=freq_bins)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_db, sr=sr, hop_length=512, x_axis="time", y_axis="log")
    plt.colorbar(format="%+2.0f dB")
    plt.title("Spectrogram (dB)")
    plt.savefig('./spec/{}.png'.format(wav_name))


    print(quantized_tokens.shape) 
    return quantized_tokens.flatten()


def fixed_binning(frequencies, amplitudes, freq_bins=10, amp_levels=5):
    """
    
    :param frequencies: 频率数组 (Hz)
    :param amplitudes:  幅度数组 (振幅值)
    :param freq_bins:   频率划分区间数量
    :param amp_levels:  幅度划分级别数量
    :return: token 序列 (list of strings)
    """

    freq_min, freq_max = np.min(frequencies), np.max(frequencies)
    freq_edges = np.linspace(freq_min, freq_max, freq_bins + 1) 
    freq_labels = [f"FREQ_{i+1}" for i in range(freq_bins)]  

    amp_min, amp_max = np.min(amplitudes), np.max(amplitudes)
    amplitudes = (amplitudes - amp_min) / (amp_max - amp_min)  
    amp_edges = np.linspace(amp_min, amp_max, amp_levels + 1)
    amp_labels = [f"AMP_{i+1}" for i in range(amp_levels)]  

    # breakpoint()

    tokens = []
    for f, a in zip(frequencies, amplitudes):
        freq_bin = np.digitize(f, freq_edges) - 1
        freq_bin = min(freq_bin, freq_bins - 1)  
        freq_token = freq_labels[freq_bin]

        amp_bin = np.digitize(a, amp_edges) - 1
        amp_bin = min(amp_bin, amp_levels - 1)  
        amp_token = amp_labels[amp_bin]

        tokens.append(f"{freq_token}_{amp_token}")

    return tokens

def steps_binning(frequencies, amplitudes, freq_bins=10, amp_levels=5):
    """
    
    :param frequencies: 频率数组 (Hz)
    :param amplitudes:  幅度数组 (振幅值)
    :param freq_bins:   频率划分区间数量
    :param amp_levels:  幅度划分级别数量
    :return: token 序列 (list of strings)
    """
    freq_min, freq_max = np.min(frequencies), np.max(frequencies)

    freq_min = freq_max / (1.2**(freq_bins-1))

    freq_edges = np.geomspace(freq_min, freq_min * 1.2**(freq_bins-1), num=freq_bins)

    freq_labels = [f"FREQ_{i+1}" for i in range(freq_bins)] 

    amp_min, amp_max = np.min(amplitudes), np.max(amplitudes)
    # print(amp_min,amp_max)

    if amp_min == amp_max:
        # breakpoint()
        amplitudes = np.zeros_like(frequencies)
        amp_edges = np.linspace(0, 1, amp_levels + 1)
    else:
        amplitudes = (amplitudes - amp_min) / (amp_max - amp_min)  

        amp_min = amp_max / (1.2**(amp_levels-1))
        amp_edges = np.geomspace(amp_min, amp_max, num=amp_levels)

    amp_labels = [f"AMP_{i+1}" for i in range(amp_levels)]  

    # breakpoint()

    tokens = []
    for f, a in zip(frequencies, amplitudes):
        freq_bin = np.digitize(f, freq_edges) - 1
        freq_bin = min(freq_bin, freq_bins - 1)  
        freq_token = freq_labels[freq_bin]

        amp_bin = np.digitize(a, amp_edges) - 1
        amp_bin = min(amp_bin, amp_levels - 1)  
        amp_token = amp_labels[amp_bin]

        tokens.append(f"{freq_token}_{amp_token}")
    
    return tokens
 
def process_wav_to_tokens(wav_file, freq_bins=10, amp_levels=5, n_fft=1024, hop_length=512, mode='steps_binning'):

    if mode == 'bin':
        y, sr = librosa.load(wav_file, sr=None)  

        D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)  
        frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)  
        magnitudes = np.abs(D) 

        # breakpoint() 

        magnitudes = magnitudes / np.max(magnitudes)  

        frame_idx = 10  
        amplitudes = magnitudes[:, frame_idx]  

        tokens = fixed_binning(frequencies,amplitudes,freq_bins=freq_bins,amp_levels=amp_levels)

    elif mode == 'encodec':
        tokens = encodec_token(wav_file)
    elif mode == 'spec_binning':
        tokens = fixed_spec_binning(wav_file)
    elif mode == 'steps_binning':

        y, sr = librosa.load(wav_file, sr=None)  

        D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)  
        frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)  
        magnitudes = np.abs(D)  

        magnitudes = magnitudes / np.max(magnitudes)  

        frame_idx = 10  
        amplitudes = magnitudes[:, frame_idx]  

        mask = frequencies < 500
        frequencies_filtered = frequencies[mask]
        amplitudes_filtered = amplitudes[mask]

        tokens = steps_binning(frequencies_filtered, amplitudes_filtered, freq_bins=freq_bins,amp_levels=amp_levels)
    else:
        print('invalid tokenizer:', mode)


    # tokenizer.add_tokens(tokens)
    # tokenizer.save_pretrained("../data/Haptic/pretrain/updated_llama_tokenizer_{}.pt".format(mode))
    # print(10*'-'+'tokenizer saved'+10*'-')

    return tokens

def build_pairwise_from_scores(json_dir, signal_dir, dpo_output, min_diff=0.5, mode='encodec'):

    data = read_json(json_dir)

    dpo_data = []
    def build_pairwise(items):
        pairwise_data = []
        for (text1, score1), (text2, score2) in combinations(items, 2):
            if abs(score1 - score2) >= min_diff:
                if score1 > score2:
                    chosen, rejected = text1, text2
                else:
                    chosen, rejected = text2, text1
                pairwise_data.append({
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected
                })
        return pairwise_data
        

    for sig, scores in data.items():
        if sig == '1228.wav':
            continue
        signal_path = os.path.join(signal_dir,'{}'.format(str(sig)))
        prompt = process_wav_to_tokens(signal_path, mode=mode)
        prompt = ' '.join(prompt)

        items = list(scores.items())

        sensory = items[:4]
        emotion = items[4:8]
        association = items[8:]

        def filtered(items, bar):
            filtered_items = [(cap, score) for cap, score in items if score >= bar]
            return filtered_items
        
        sensory = filtered(sensory, bar=3.5)
        emotion = filtered(emotion, bar=3.5)
        association = filtered(association, bar=3.5)

        sen_pairwise = build_pairwise(sensory)
        emo_pairwise = build_pairwise(emotion)
        ass_pairwise = build_pairwise(association)

        dpo_data.append({
            'sen':sen_pairwise,
            'emo':emo_pairwise,
            'ass':ass_pairwise})
    
    to_json(dpo_data, dpo_output)
    print('completed dpo data')

def build_pairwise_from_scores_with_filter(json_dir, signal_dir, dpo_output, min_diff=0.5, mode='encodec'):

    data = read_json(json_dir)

    dpo_data = []
    def build_pairwise(items):
        pairwise_data = []
        for (text1, score1), (text2, score2) in combinations(items, 2):
            if abs(score1 - score2) >= min_diff:
                if score1 > score2:
                    chosen, rejected = text1, text2
                else:
                    chosen, rejected = text2, text1
                pairwise_data.append({
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected
                })
        return pairwise_data
        

    for sig, scores in data.items():
        print(sig)
        if sig == '1228.wav':
            continue
        signal_path = os.path.join(signal_dir,'{}'.format(str(sig)))
        prompt = process_wav_to_tokens(signal_path, mode=mode)
        prompt = ' '.join(prompt)

        items = list(scores.items())

        sensory = items[:4]
        emotion = items[4:8]
        association = items[8:]

        breakpoint()

        ##filter out the captions their scores lower than 3.

        sen_pairwise = build_pairwise(sensory)
        emo_pairwise = build_pairwise(emotion)
        ass_pairwise = build_pairwise(association)

        dpo_data.append({
            'sen':sen_pairwise,
            'emo':emo_pairwise,
            'ass':ass_pairwise})
    
    to_json(dpo_data, dpo_output)
    print('completed dpo data')


json_dir = r"../data/Haptic/signals/ratings/combined_rating.json"
signal_dir = r"../data/Haptic/signals/test_signals"
dpo_output = r'../data/Haptic/jsons/dpo_data_filtered.json'
build_pairwise_from_scores(json_dir, signal_dir, dpo_output)