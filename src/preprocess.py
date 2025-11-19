from wave import WAVE_FORMAT_PCM
import numpy as np
import scipy.io.wavfile as wav
import librosa
import librosa.display
import matplotlib.pyplot as plt
import torch
import torchaudio
import os
import random
import yaml
from datasets import Dataset, Audio
from utils.util import read_json,to_pickle,to_json
from transformers import LlamaTokenizer,AutoTokenizer,AutoProcessor,EncodecModel
# from repcodec.RepCodec import RepCodec
from utils.feature_extractor import Feature_Extractor
from config import Config


encodec_model = EncodecModel.from_pretrained("facebook/encodec_24khz")
processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")
access_token = ''
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B",token=access_token)
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B",token=access_token)
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


def repcodec_token(wav_file):
    with open(config) as fp:
        conf = yaml.load(fp, Loader=yaml.FullLoader)

    extractor = Feature_Extractor(extractor_name='', sample_rate=16000)

    model = RepCodec(**conf)
    model.load_state_dict(torch.load(repcodec_trained_model_path, map_location="cpu")["model"]["repcodec"])
    model.quantizer.initial()
    model.eval()

    haptic_set = [wav_file]
    data_dict = {"audio": haptic_set}

    data_dataset = Dataset.from_dict(data_dict).cast_column("audio", Audio())
    data_size = len(data_dataset)

    for i in range(data_size): 
        audio = data_dataset[i]["audio"]["array"]
        haptic_features = extractor(audio)
        print('haptic_feature.shape:{}'.format(haptic_features.shape))

        with torch.no_grad():
            x = model.encoder(haptic_features)
            z = model.projector(x)
            _, idx = model.quantizer.codebook.forward_index(z.transpose(2, 1))
            tokens = idx.cpu().data.numpy().tolist()[0]
    
    return tokens

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
    """
    读取 .wav 文件并转换为固定区间分桶的 token 序列。

    :param wav_file: 输入 .wav 文件路径
    :return: token 序列
    """
    if mode == 'bin':
        # frequency_dom(wav_file=wav_file)
        # frequency_dom_former(wav_file=wav_file)
        y, sr = librosa.load(wav_file, sr=None)  

        D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)  
        frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)  
        magnitudes = np.abs(D) 

        # breakpoint() 

        magnitudes = magnitudes / np.max(magnitudes)  

        frame_idx = 10  
        amplitudes = magnitudes[:, frame_idx]  

        # mask = frequencies < 500
        # frequencies_filtered = frequencies[mask]
        # amplitudes_filtered = amplitudes[mask]

        tokens = fixed_binning(frequencies,amplitudes,freq_bins=freq_bins,amp_levels=amp_levels)

        # tokens = fixed_binning(frequencies_filtered,amplitudes_filtered,freq_bins=freq_bins,amp_levels=amp_levels)

    elif mode == 'encodec':
        # breakpoint()
        tokens = encodec_token(wav_file)

    elif mode == 'repcodec':
        tokens = repcodec_token(wav_file)
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

def read_haptics(haptic_path):
    haptic_list = []

    all_signals = read_json(haptic_path)

    for signal_id in all_signals.keys():
        haptic_list.append(signal_id)

    return haptic_list

def split_data(signal_path, json_path, aug=False, mode='steps_binning'):

    haptic_list = read_haptics(haptic_path=json_path)

    train_size = 0.8
    val_size = 0.1
    test_size = 0.1
    
    indices = np.random.permutation(len(haptic_list))

    train_end = int(train_size * len(haptic_list))
    val_end = train_end + int(val_size * len(haptic_list))
    

    train_haptic = [haptic_list[i] for i in indices[:train_end]]
    valid_haptic = [haptic_list[i] for i in indices[train_end:val_end]]
    test_haptic = [haptic_list[i] for i in indices[val_end:]]

    print("length of train:", len(train_haptic))
    print("length of valid:", len(valid_haptic))
    print("length of test:", len(test_haptic))


    all_signals = read_json(json_path)

    train_data, train_ground = generate_dataset(signal_dir=signal_path, all_signals=all_signals, cur_keys=train_haptic, augmented=aug, mode=mode)
    valid_data, valid_ground = generate_dataset(signal_dir=signal_path, all_signals=all_signals, cur_keys=valid_haptic, augmented=aug, mode=mode)
    test_data, test_ground = generate_dataset(signal_dir=signal_path, all_signals=all_signals, cur_keys=test_haptic, augmented=aug, mode=mode)
    
    ##just for test
    # test_signal = r'../data/Haptic/signals/test_signals/'
    # test_signal_list  = list(range(273, 682))
    # sep_test_data = process_test(test_signal, test_signal_list)
    ##just for test

    # merged_dict = train_ground.copy()  # 避免修改原字典
    # merged_dict.update(valid_ground)
    # merged_dict.update(test_ground)

    # if not aug:
    #     suffix = ''
    # else:
    #     suffix = '_aug'
    # # to_json(merged_dict, '../data/Haptic/ground_data.json')

    # if mode == 'steps_binning':

    #     to_pickle(train_data, '../data/Haptic/pkl/Haptic/updated/train_steps_binning_haptic_5_1{}.pkl'.format(suffix))
    #     to_pickle(valid_data, '../data/Haptic/pkl/Haptic/updated/valid_steps_binning_haptic_5_1{}.pkl'.format(suffix))
    #     to_pickle(test_data, '../data/Haptic/pkl/Haptic/updated/test_steps_binning_haptic_5_1{}.pkl'.format(suffix))

    # elif mode == 'encodec':
    #     to_pickle(train_data, '../data/Haptic/pkl/Haptic/updated/train_encodec_haptic_5_1{}.pkl'.format(suffix))
    #     to_pickle(valid_data, '../data/Haptic/pkl/Haptic/updated/valid_encodec_haptic_5_1{}.pkl'.format(suffix))
    #     to_pickle(test_data, '../data/Haptic/pkl/Haptic/updated/test_encodec_haptic_5_1{}.pkl'.format(suffix))

    # else:
    #     print('invalid tokenizer')


    ## just for test
    # to_pickle(sep_test_data, '../data/Haptic/pkl/Haptic/updated/test_step.pkl')
    ## just for test



def process_test(signal_dir, cur_keys):

    data = []
    for signal_id in cur_keys:
        signal_path = os.path.join(signal_dir,'F{}_loop.wav'.format(str(signal_id)))
        if not os.path.exists(signal_path):
            signal_path = os.path.join(signal_dir,'F{}.wav'.format(str(signal_id)))
        
        print(signal_path)
        if os.path.exists(signal_path):
            tokens = process_wav_to_tokens(signal_path)
            sensory_sample = [signal_id, tokens, '', 'sensory']
            emotion_sample = [signal_id, tokens, '', 'emotion']
            association_sample = [signal_id, tokens, '', 'association']

            data.append(sensory_sample)
            data.append(emotion_sample)
            data.append(association_sample)
    
    # random.shuffle(data)
    # random.shuffle(data)
    # random.shuffle(data)
    print('length of test data:{}'.format(len(data)))

    return data

def generate_dataset(signal_dir, all_signals, cur_keys, augmented=False, mode='steps_binning'):

    data = []
    ground_data = {}
    token_summary = []
    
    sen_data, emo_data, ass_data = [], [], []
    # print(type(all_signals))
    for signal_id in cur_keys:
        user_ids = all_signals[signal_id].keys()
        signal_path = os.path.join(signal_dir,'F{}_loop.wav'.format(str(signal_id)))
        if not os.path.exists(signal_path):
            signal_path = os.path.join(signal_dir, 'F{}.wav'.format(str(signal_id)))
        # print(signal_path)
        tokens = process_wav_to_tokens(signal_path, mode=mode)
        token_summary.append(len(tokens))

        if signal_id not in ground_data.keys(): 
            ground_data[signal_id] = {'sensory':[], 'emotion':[], 'association':[]}
        else:
            print('some problem occurs')
        for uid in user_ids:
            if uid != 'vibviz':
                desc = all_signals[signal_id][uid]

                sensory = desc['free_text_sensory']
                emotion = desc['free_text_emotional']
                if 'free_text_association' in desc:
                    association = desc['free_text_association']
                else:
                    association = 'N.A'

                ground_data[signal_id]['sensory'].append(sensory)
                ground_data[signal_id]['emotion'].append(emotion)
                ground_data[signal_id]['association'].append(association)


                sensory_sample = [signal_id, tokens, sensory, 'sensory']
                emotion_sample = [signal_id, tokens, emotion, 'emotion']
                association_sample = [signal_id, tokens, association, 'association']

                sen_data.append(sensory_sample)
                emo_data.append(emotion_sample)
                ass_data.append(association_sample)

                data.append(sensory_sample)
                data.append(emotion_sample)
                data.append(association_sample)

            else:
                pass

        if augmented:
            user_ids = all_signals[signal_id].keys()

            for i in range(8):
                aug_signal_path = os.path.join(signal_dir,'F{}_loop_aug{}.wav'.format(str(signal_id), i))
                if not os.path.exists(aug_signal_path):
                    aug_signal_path = os.path.join(signal_dir, 'F{}_aug{}.wav'.format(str(signal_id), i))
                # print(aug_signal_path)
                aug_tokens = process_wav_to_tokens(aug_signal_path,mode=mode)
                token_summary.append(len(aug_tokens))

                for uid in user_ids:
                    if uid != 'vibviz':
                        desc = all_signals[signal_id][uid]

                        sensory = desc['free_text_sensory']
                        emotion = desc['free_text_emotional']
                        if 'association' in desc:
                            association = desc['free_text_association']

                        ground_data[signal_id]['sensory'].append(sensory)
                        ground_data[signal_id]['emotion'].append(emotion)
                            
                        ground_data[signal_id]['association'].append(association)


                        sensory_sample = [signal_id, aug_tokens, sensory, 'sensory']
                        emotion_sample = [signal_id, aug_tokens, emotion, 'emotion']
                        association_sample = [signal_id, aug_tokens, association, 'association']

                        sen_data.append(sensory_sample)
                        emo_data.append(emotion_sample)
                        ass_data.append(association_sample)

                        data.append(sensory_sample)
                        data.append(emotion_sample)
                        data.append(association_sample) 

                    else:
                        pass

    random.shuffle(data)
    random.shuffle(data)
    random.shuffle(data)

    avg_len = np.mean(token_summary)
    max_len = np.max(token_summary)
    min_len = np.min(token_summary)

    print('avg_len:{},max_len:{}, min_len:{}'.format(avg_len, max_len, min_len))

    print('length of data:{}'.format(len(data)))

    print('length of sensory:{}, length of emotion:{}, length of association:{}'.format(len(sen_data), len(emo_data), len(ass_data)))

    return data, ground_data



signal_dir = "../data/Haptic/signals/aug_signal"
json_dir = "../data/Haptic/jsons/signal_map_5.1.json"
curr = 'steps_binning'
if curr == 'encodec':
    mode = 'encodec'
    aug = False
elif curr == 'steps_binning':
    mode = 'steps_binning'
    aug = True
else:
    print('invalid tokenizer')

split_data(signal_path=signal_dir, json_path=json_dir, aug=aug, mode=mode)    