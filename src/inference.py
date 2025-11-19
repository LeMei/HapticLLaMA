global logger

import re
from tkinter import NO
from matplotlib.dates import date2num
import torch
from torch import nn
import librosa
import torch.optim as optim
import numpy as np
import os
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datasets import Dataset, Audio
from transformers import LlamaTokenizer,AutoTokenizer,AutoProcessor,EncodecModel,GPT2Tokenizer, GPT2LMHeadModel
from transformers import T5ForConditionalGeneration, AutoTokenizer
import openai

from pathlib import Path
import re
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import itertools

from config import get_args, get_config, DEVICE
from model import Model
from utils.util import calculate_similarity
from config import get_args, get_config
from utils.util import read_json,to_pickle,to_json


spell_model = T5ForConditionalGeneration.from_pretrained("ai-forever/T5-large-spell")
spell_tokenizer = AutoTokenizer.from_pretrained("ai-forever/T5-large-spell")
prefix = "grammar: "

gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt_model = GPT2LMHeadModel.from_pretrained('gpt2')
gpt_model.eval()

encodec_model = EncodecModel.from_pretrained("facebook/encodec_24khz")
processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")

step_tokenizer = AutoTokenizer.from_pretrained("../data/Haptic/pretrain/updated_llama_tokenizer_steps_binning.pt")
encodec_tokenizer = AutoTokenizer.from_pretrained("../data/Haptic/pretrain/updated_llama_tokenizer_encodec.pt")

epoch = 2
output_dir = './'
version='40_vib'
date = '5.30'
caption_log = r'./captions/two_models_inference_caption_{}_{}.txt'.format(version,date)
full_caption_log = r'./captions/two_models_inference_full_caption_{}_{}.txt'.format(version,date)
bar = 0.5
args = get_args()
N=4

def is_complete_caption(caption):
    if len(caption.split()) < 5:
        return False
    if not re.search(r'[.?!]$', caption):
        return False
    if re.search(r'\b(and|but|with|because|then|while|although|as)\s*$', caption):
        return False
    return True

def llm_score(caption):

    def compute_perplexity(sentence):
        inputs = gpt_tokenizer(sentence, return_tensors="pt")
        with torch.no_grad():
            outputs = gpt_model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
        return torch.exp(loss).item()
    

    ppl = compute_perplexity(caption)
    # print("{}:{}".format(caption,ppl))

    return ppl


def check_sentence_completeness(sentence):
    prompt = f"Is the following sentence complete and grammatically correct? Reply 'yes' or 'no'.\n\n\"{sentence}\""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response['choices'][0]['message']['content'].strip().lower()

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

    freq_min, freq_max = np.min(frequencies), np.max(frequencies)
    freq_edges = np.linspace(freq_min, freq_max, freq_bins + 1) 
    freq_labels = [f"FREQ_{i+1}" for i in range(freq_bins)]  

    amp_min, amp_max = np.min(amplitudes), np.max(amplitudes)
    amplitudes = (amplitudes - amp_min) / (amp_max - amp_min)  
    amp_edges = np.linspace(amp_min, amp_max, amp_levels + 1)
    amp_labels = [f"AMP_{i+1}" for i in range(amp_levels)]  


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

def process_wav_to_tokens(wav_file, freq_bins=10, amp_levels=5, n_fft=1024, hop_length=512, mode='steps_binning'):
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
    elif mode == 'encodec':
        tokens = encodec_token(wav_file)
    else:
        print('invalid tokenizer:', mode)
    return tokens


def load_model(epoch, device, output_dir, version, mode, model_file_url=None):
        if model_file_url is None or len(model_file_url) == 0:
            model_file_url = os.path.join(output_dir, "full_lora_{}_{}.pth".format(epoch, version))
        if os.path.exists(model_file_url):
            model = Model(args, mode=mode)

            lora_state_dict = torch.load(model_file_url)
            state_name, model_name = [], []
            # print('--------------------original**model-------------------')
            for name, param in model.named_parameters():
                model_name.append(name)
            # print('--------------------state**dict-------------------')
            for name in lora_state_dict.keys():
                state_name.append(name)
            print('len1:{},len2:{}'.format(len(model_name), len(state_name)))
            missing_keys, unexpected_keys = model.load_state_dict(lora_state_dict, strict=False)
            # print("Missing keys:", missing_keys)
            # print("Unexpected keys:", unexpected_keys)
            model.to(device)
        else:
            model = None
        return model

def tokenizer_haptic(haptic, category):

    def formalize_input(haptic_tokens, tokenizer, category):
        prompts = "the {} description is:".format(category.strip())
        tokenizer.pad_token = tokenizer.eos_token

        inputs = tokenizer(haptic_tokens, padding=True, truncation=True, return_tensors="pt")
        input_ids = inputs.input_ids
        input_atts = inputs.attention_mask

        prompt_enc = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt")
        prompt_ids = prompt_enc.input_ids
        prompt_atts = prompt_enc.attention_mask

        prompt_ids = torch.cat((input_ids,prompt_ids),dim=1)
        prompt_atts = torch.cat((input_atts,prompt_atts),dim=1)


        return input_ids,input_atts, prompt_ids, prompt_atts

    step_haptic_tokens = process_wav_to_tokens(haptic, mode='steps_binning')
    step_haptic_tokens = [' '.join(step_haptic_tokens)]

    step_input_ids,step_input_atts, step_prompt_ids, step_prompt_atts = formalize_input(step_haptic_tokens, step_tokenizer, category)


    encodec_haptic_tokens = process_wav_to_tokens(haptic, mode='encodec')
    encodec_haptic_tokens = [' '.join(encodec_haptic_tokens)]

    encodec_input_ids, encodec_input_atts, encodec_prompt_ids, prompt_atts = formalize_input(encodec_haptic_tokens, encodec_tokenizer, category)

    return (step_input_ids, step_input_atts, step_prompt_ids, step_prompt_atts), (encodec_input_ids, encodec_input_atts, encodec_prompt_ids, prompt_atts) 

def inference_with_one_haptic_times(haptic_files, epoch, device, output_dir, version, model_file_url=None, T=6):

    model = load_model(epoch=epoch, device=DEVICE, output_dir=output_dir, version=version, model_file_url=model_file_url)
    model.eval()
    f = open(caption_log, 'w')
    
    for file_path in Path(haptic_files).rglob('F*_loop*.wav'):
        print(file_path)
        sid = file_path.name
        for category in ['sensory', 'emotion', 'association']:  
            (input_ids,input_atts), (prompt_ids, prompt_atts) = tokenizer_haptic(file_path, category)

            with torch.cuda.device(0):
                prompt_ids, prompt_atts = prompt_ids.to(DEVICE), prompt_atts.to(DEVICE)

            print('---------------------------signal id:{}, category:{}------------------------'.format(sid, category))
            f.write('---------------------------signal id:{}, category:{}------------------------\n'.format(sid, category))
            with torch.no_grad():
                multi_caption_list = []

                for t in range(T):
                    generated_tokens = model.generate(inputs = prompt_ids,input_atts=prompt_atts)
                    generated_caption = generated_tokens[0].split('.')[0]+'.'
                    
                    multi_caption_list.append(generated_caption)

            avg_score,row_similarity, cosine_caption = calculate_similarity(multi_caption_list)
            print('avg similarity score:{}'.format(avg_score))

            pairs = list(itertools.combinations(range(T), 2))
            pairs.sort(key=lambda x: cosine_caption[x[0]][x[1]], reverse=False)

            filtered_set = []
            for idx in range(3):
                i, j = pairs[idx]
                print(f"{multi_caption_list[i]} <--> {multi_caption_list[j]} | Distance: {cosine_caption[i][j]:.3f}")
                f.write(f"{multi_caption_list[i]} <--> {multi_caption_list[j]} | Distance: {cosine_caption[i][j]:.3f}")
                f.write('\n')

                filtered_set.append(multi_caption_list[i])
                filtered_set.append(multi_caption_list[j])
            
            filtered_set = set(filtered_set)

            for i, caption in enumerate(filtered_set):
                print('{}:{}'.format(i, caption))
                f.write('{}:{}\n'.format(i, caption))

def extract_caption(generated_tokens):
    token = generated_tokens
    match = re.search(r"<eos>?|<eo", token)
    if match:
        generated_caption = token[:match.start()].strip()
    elif '.' in generated_tokens:
        generated_caption = generated_tokens.split('.')[0].strip()+'.'
    else:
        generated_caption = generated_tokens.split('and')[0].strip()+'.'

    if not generated_caption.endswith("."):
        generated_caption = generated_caption.strip() + '.'

    return generated_caption 

    
def inference_with_one_haptic_models(haptic_files, epoch, device, output_dir, version, step_model_file_url=None, encodec_model_file_url=None,T=6, output_file=None):


    step_model = load_model(epoch=epoch, device=DEVICE, output_dir=output_dir, version=version, mode='steps_binning', model_file_url=step_model_file_url)
    step_model.eval()
    print('load step model')

    encodec_model = load_model(epoch=epoch, device=DEVICE, output_dir=output_dir, version=version, mode='encodec', model_file_url=encodec_model_file_url)
    encodec_model.eval()
    print('load encodec model')
    if not output_file:
        f = open(caption_log, 'w')
        full_f = open(full_caption_log, 'w')
    else:
        f = open(output_file, 'a+')
        full_f = open(full_caption_log, 'w')
    for file_path in Path(haptic_files).rglob('F*.wav'):
        sid = file_path.name
        filename = file_path.name
        if '_loop' in filename:
            continue
        print(file_path)
        for category in ['sensory', 'emotion', 'association']:  
            (step_input_ids, step_input_atts, step_prompt_ids, step_prompt_atts), (encodec_input_ids, encodec_input_atts, encodec_prompt_ids, encodec_prompt_atts) \
            = tokenizer_haptic(str(file_path), category)

            with torch.cuda.device(0):
                step_prompt_ids, step_prompt_atts = step_prompt_ids.to(DEVICE), step_prompt_atts.to(DEVICE)
                encodec_prompt_ids, encodec_prompt_atts = encodec_prompt_ids.to(DEVICE), encodec_prompt_atts.to(DEVICE)

            print('---------------------------signal id:{}, category:{}------------------------'.format(sid, category))
            f.write('---------------------------signal id:{}, category:{}------------------------\n'.format(sid, category))
            full_f.write('---------------------------signal id:{}, category:{}------------------------\n'.format(sid, category))

            with torch.no_grad():
                multi_caption_list = []
                step_full_caption_list = []
                encodec_full_caption_list = []
                for t in range(T):
                    ### step
                    step_generated_tokens = step_model.generate(inputs = step_prompt_ids,input_atts=step_prompt_atts)
                    step_generated_caption = extract_caption(step_generated_tokens[0])

                    # print('step:', step_generated_caption)

                    step_full_caption_list.append(step_generated_caption)
                    ##score
                    # if is_complete_caption(step_generated_caption) and check_sentence_completeness(step_generated_caption):
                    #     multi_caption_list.append(step_generated_caption)
                    if is_complete_caption(step_generated_caption) and llm_score(step_generated_caption)>0.5:
                        multi_caption_list.append(step_generated_caption)

                    if len(multi_caption_list) == 4:
                        break
                    ## step
                
                if len(multi_caption_list) < 4:
                    RT = 2*(4 - len(multi_caption_list))
                    for t in range(RT):
                        ### step
                        step_generated_tokens = step_model.generate(inputs = step_prompt_ids,input_atts=step_prompt_atts)
                        step_generated_caption = extract_caption(step_generated_tokens[0])
                        # print('step:', step_generated_caption)
                        step_full_caption_list.append(step_generated_caption)
                        ##score
                        # ppl = llm_score(step_generated_caption)
                        # if is_complete_caption(step_generated_caption) and check_sentence_completeness(step_generated_caption):
                        #     multi_caption_list.append(step_generated_caption)
                        if is_complete_caption(step_generated_caption) and llm_score(step_generated_caption)>0.5:
                            multi_caption_list.append(step_generated_caption)
                        
                        if len(multi_caption_list) == 4:
                            break

                for t in range(T):
                    ## encodec 
                    encodec_generated_tokens = encodec_model.generate(inputs = encodec_prompt_ids,input_atts=encodec_prompt_atts)

                    encodec_generated_caption = extract_caption(encodec_generated_tokens[0])

                    # print('encodec:', encodec_generated_caption)    
                    encodec_full_caption_list.append(encodec_generated_caption)

                    # ppl = llm_score(step_generated_caption)

                    # if is_complete_caption(encodec_generated_caption) and check_sentence_completeness(encodec_generated_caption):
                    #     multi_caption_list.append(encodec_generated_caption)

                    if is_complete_caption(encodec_generated_caption) and llm_score(encodec_generated_caption)>0.5:
                        multi_caption_list.append(encodec_generated_caption)
                    
                    if len(multi_caption_list) == 8:
                        break
                    ## encodec 

                if len(multi_caption_list) < 8:
                    RT = 2*(8 - len(multi_caption_list))
                    for t in range(RT):
                        ### step
                        encodec_generated_tokens = encodec_model.generate(inputs = encodec_prompt_ids,input_atts=encodec_prompt_atts)
                        encodec_generated_caption = extract_caption(encodec_generated_tokens[0])
                        # print('encodec:', encodec_generated_caption)    

                        encodec_full_caption_list.append(encodec_generated_caption)
                        ##score
                        # if is_complete_caption(encodec_generated_caption) and check_sentence_completeness(encodec_generated_caption):
                        #     multi_caption_list.append(encodec_generated_caption)

                        if is_complete_caption(encodec_generated_caption) and llm_score(encodec_generated_caption)>0.5:
                            multi_caption_list.append(encodec_generated_caption)
                        
                        if len(multi_caption_list) == 8:
                            break
                

    #         avg_score,row_similarity, cosine_caption = calculate_similarity(multi_caption_list)
    #         print('avg similarity score:{}'.format(avg_score))

    #         pairs = list(itertools.combinations(range(len(multi_caption_list)), 2))

    #         for x in pairs:
    #             if x[0] >= cosine_caption.shape[0] or x[1] >= cosine_caption.shape[1]:
    #                 print(f"Invalid index pair: {x}")

    #         pairs.sort(key=lambda x: cosine_caption[x[0]][x[1]], reverse=False)
    #         # breakpoint()
    #         filtered_set = []
    #         for idx in range(6):
    #             i, j = pairs[idx]
    #             print(f"{multi_caption_list[i]} <--> {multi_caption_list[j]} | Distance: {cosine_caption[i][j]:.3f}")
    #             f.write(f"{multi_caption_list[i]} <--> {multi_caption_list[j]} | Distance: {cosine_caption[i][j]:.3f}")
    #             f.write('\n')

    #             filtered_set.append(multi_caption_list[i])
    #             filtered_set.append(multi_caption_list[j])
            
    #         print('filtered_set:{}'.format(len(filtered_set)))

    #         filtered_set = list(set(filtered_set))[:N]
    #         print('after_filtering:{}'.format(len(filtered_set)))

    #         for i, caption in enumerate(filtered_set):
    #             print('{}:{}'.format(i, caption))
    #             f.write('{}:{}\n'.format(i, caption))

    #         for i, caption in enumerate(step_full_caption_list):
    #             print('step: {}:{}'.format(i, caption))
    #             # full_f.write('step: {}:{}\n'.format(i, caption))

    #         for i, caption in enumerate(encodec_full_caption_list):
    #             print('encodec: {}:{}'.format(i, caption))
    #             full_f.write('encodec: {}:{}\n'.format(i, caption))

    # f.close()
    # full_f.close()


# haptic_files = r'../data/Haptic/signals/test_signals/'
# test_vib = r'../data/Haptic/signals/40_vib/'

step_model_url = r'./pretrained_model/step/steps_binning_full_lora_2_5.1_aug.pth'
encodec_model_url = r'./pretrained_model/encodec/encodec_full_lora_2_5.1.pth'

count_set = 22
vib_sets = list(range(0, count_set))
# vib_sets = [15,16,17,18,21]
haptic_dir = r'../data/Haptic/signals/sets/'

for i in vib_sets:
    output_f = r'../data/Haptic/signals/captions/set{}_caption.txt'.format(i)
    test_vib = haptic_dir + str(i)
    inference_with_one_haptic_models(haptic_files=test_vib, epoch=epoch, device=DEVICE, output_dir=output_dir, version=version,step_model_file_url=step_model_url, encodec_model_file_url=encodec_model_url,output_file=output_f)




            

                                


            




            

    

    
