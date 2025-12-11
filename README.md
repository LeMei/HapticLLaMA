# 📌 HapticLLaMA: A Multimodal Sensory Language Model for Haptic Captioning

Arxiv: https://arxiv.org/pdf/2508.06475?

---

## 📖 Introduction
**HapticLLaMA** is a multimodal sensory language model that interprets vibration signals into descriptions in a given sensory, emotional, or associative category. 
HapticLLaMA is trained in two stages: (1) supervised fine-tuning using the LLaMA architecture with LoRA-based adaptation, and (2) fine-tuning via reinforcement
learning from human feedback (RLHF). 

---

## 🧩 Tasks
- Given a vibration signal S and a target category c ∈ {sensory, emotional, associative}, where sensory refers to physical attributes (e.g.,intensity of tapping), emotional denotes affective
impressions (e.g., the mood of a scene), and associative indicates real-world familiar experiences (e.g., buzzing of a bee, a heartbeat), the goal is to generate a caption corresponding to the specified category of haptic experience.


## 📂 Training
HapticLLaMA training is consist of (1) supervised fine-tuning with LoRA adaptation and (2) subsequent fine-tuning based on human feedback on generated captions.

<img width="925" height="557" alt="image" src="https://github.com/user-attachments/assets/28a0aa75-d011-4870-b9ec-b9b3607eb8d8" />


- ## 📂 Models
  Also can be found in https://huggingface.co/GuiminHu/HapticLLaMA
- **Frequency-based Model** (frequency_llama.pt):

  https://drive.google.com/file/d/1qF5KEcjNHdQ2AUlwr1mZzW1EHl-rHfqh/view?usp=sharing
  
- **Encodec-based Model** (encodec_llama.pt):

  https://drive.google.com/file/d/1RMC6mgNHPDm_m6emHNTElNS8k6EHwiJ5/view?usp=sharing

- **Haptic tokenizer**

  Encodec-based tokenizer: https://drive.google.com/drive/folders/1Fd4FWbjqjje8pk2rZvRnrl1qwuXUcM46?usp=sharing

  Frequency-based tokenizer: https://drive.google.com/drive/folders/1N-rgfn7-Rm_ebWxYwO5UBUMOSoT0cd1z?usp=sharing

- ## 📂 VibRate
  https://drive.google.com/drive/folders/1rxAOkuZAZ_BNAHwzggWTBpXVrA0UwJz1?usp=sharing

  or can be found in: https://huggingface.co/datasets/GuiminHu/VibRate

## 📂 Haptic Tokenizer
- **Frequency-based Tokenizer**:
  
  <img width="361" height="211" alt="image" src="https://github.com/user-attachments/assets/ca848d0b-18d5-4ad5-89e4-268399aad801" />

Frequency-based Tokenizer divides the frequency range into logarithmically spaced bins that correspond to just-noticeable ifferences in human frequency perception. Similarly, the amplitude range is segmented into normalized levels. The tokenizer then assigns a unique
token (e.g., FREQ_3_AMP_2) to each frequencyamplitude pair, encoding the signal’s spectral content into a form interpretable by LLMs.
```python
import librosa

def steps_binning(frequencies, amplitudes, freq_bins=10, amp_levels=5):

    freq_min, freq_max = np.min(frequencies), np.max(frequencies)
    freq_min = freq_max / (1.2**(freq_bins-1))
    freq_edges = np.geomspace(freq_min, freq_min * 1.2**(freq_bins-1), num=freq_bins)
    freq_labels = [f"FREQ_{i+1}" for i in range(freq_bins)] 
    amp_min, amp_max = np.min(amplitudes), np.max(amplitudes)
    if amp_min == amp_max:
        # breakpoint()
        amplitudes = np.zeros_like(frequencies)
        amp_edges = np.linspace(0, 1, amp_levels + 1)
    else:
        amplitudes = (amplitudes - amp_min) / (amp_max - amp_min)  
        amp_min = amp_max / (1.2**(amp_levels-1))
        amp_edges = np.geomspace(amp_min, amp_max, num=amp_levels)

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

  ### start load .wav file and tokenize
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
  ###haptic tokens based on Frequency-base haptic tokenizer
  tokens = steps_binning(frequencies_filtered, amplitudes_filtered, freq_bins=freq_bins,amp_levels=amp_levels)

```
  
- **EnCodec-based Tokenizer**:

<img width="317" height="172" alt="image" src="https://github.com/user-attachments/assets/35e50d2e-c21f-4fc1-8953-74305a752ee0" />

EnCodec is a neural audio codec that compresses audio using deep learning (Défossez et al., 2023). It consists of three
main components: (1) an encoder that transforms raw audio into a lower-dimensional latent representation, (2) a quantizer that discretizes the latent features via residual vector quantization, and (3) a decoder that reconstructs the waveform from the quantized codes. EnCodec-based tokenizer extract the codes from residual vector quantization in the audio compression architecture. 
  
```python
from transformers import AutoTokenizer,AutoProcessor,EncodecModel
from datasets import Dataset, Audio

encodec_model = EncodecModel.from_pretrained("facebook/encodec_24khz")
processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")

### EnCodec-based Tokenizer
def encodec_token(wav_file):
    data_dict = {"audio": [wav_file]}
    data_dataset = Dataset.from_dict(data_dict).cast_column("audio", Audio())
    audio_sample = data_dataset[-1]["audio"]["array"]
    inputs = processor(raw_audio=audio_sample, sampling_rate=24000, return_tensors="pt")
    with torch.no_grad():
        encoded_frames = encodec_model.encode(inputs["input_values"], inputs["padding_mask"])
    tokens = encoded_frames.audio_codes[0][0]
    tokens_list = [str(token) for token in tokens[0].tolist()]

    return tokens_list
```

## 📂 Inference

Given a haptic signal, we can prompt HapticLLaMA to generate captions from sensory, emotional, and associative perspectives respectively.

<img width="448" height="329" alt="image" src="https://github.com/user-attachments/assets/2ea17083-5da3-47f2-9781-7f17912d08cc" />


```python
import torch
from torch import nn
import librosa

#load model--HapticLLaMA
def load_model(stage, device, mode, model_file_url):
        if os.path.exists(model_file_url):
            model = Model(args, mode=mode)
            lora_state_dict = torch.load(model_file_url)
            state_name, model_name = [], []
            for name, param in model.named_parameters():
                model_name.append(name)
            for name in lora_state_dict.keys():
                state_name.append(name)
            missing_keys, unexpected_keys = model.load_state_dict(lora_state_dict, strict=False)
            model.to(device)
        else:
            print('invalid model url!')
            model = None
        return model

###load pretrained haptic tokenizer

frequency_tokenizer = AutoTokenizer.from_pretrained(r"./updated_llama_tokenizer_frequency.pt/")
encodec_tokenizer = AutoTokenizer.from_pretrained(r"./updated_llama_tokenizer_encodec.pt/")

#formalize input for inference
def tokenizer_haptic(haptic, prompt, mode):

    def formalize_input(haptic_tokens, tokenizer, prompt):
        tokenizer.pad_token = tokenizer.eos_token

        inputs = tokenizer(haptic_tokens, padding=True, truncation=True, return_tensors="pt")
        input_ids = inputs.input_ids
        input_atts = inputs.attention_mask

        prompt_enc = tokenizer(prompt, padding=True, truncation=True, return_tensors="pt")
        prompt_ids = prompt_enc.input_ids
        prompt_atts = prompt_enc.attention_mask

        prompt_ids = torch.cat((input_ids,prompt_ids),dim=1)
        prompt_atts = torch.cat((input_atts,prompt_atts),dim=1)


        return input_ids,input_atts, prompt_ids, prompt_atts

    ###Frequency-based token formalization
    if mode == 'frequency':
      freq_haptic_tokens = frequency_tokenizer(haptic, mode='frequency)
      freq_haptic_tokens = [' '.join(freq_haptic_tokens)]
      freq_input_ids,freq_input_atts, freq_prompt_ids, freq_prompt_atts = formalize_input(freq_haptic_tokens, frequency_tokenizer, prompt=prompt)
      return freq_input_ids, freq_input_atts, freq_prompt_ids, freq_prompt_atts
    elif mode == 'encodec':
      ###Encodec-based token formalization
      encodec_haptic_tokens = encodec_token(haptic, mode='encodec')
      encodec_haptic_tokens = [' '.join(encodec_haptic_tokens)]
      encodec_input_ids, encodec_input_atts, encodec_prompt_ids, prompt_atts = formalize_input(encodec_haptic_tokens, encodec_tokenizer, prompt=prompt)
      return encodec_input_ids, encodec_input_atts, encodec_prompt_ids, prompt_atts
    else:
      print('invalid tokenizer...')

```
Inference for one sample

```python
haptic_signal = r'./F211_loop.wav'
sensory_prompt = 'its sensory description is'
encodec_model_file_url = r'./encodec_hapticllama.pt'
##for emotional and associative
##emotional_prompt = 'its emotional description is'
##associative_prompt = 'its associative description is'
input_ids, input_atts, prompt_ids, prompt_atts = tokenizer_haptic(haptic_signal, sensory_prompt, mode='encodec')
hapticllama = load_model(stage=1, device='cuda', mode='encodec', model_file_url=encodec_model_file_url)
caption = hapticllama.generate(inputs = prompt_ids,input_atts=prompt_atts)
print(caption)
```
---

## 🚀 Citation
If you find this work (e.g., model or code) useful for your research, please cite our paper:

```bibtex
@article{hu2025hapticllama,
  title={HapticLLaMA: A Multimodal Sensory Language Model for Haptic Captioning},
  author={Hu, Guimin and Hershcovich, Daniel and Seifi, Hasti},
  journal={arXiv preprint arXiv:2508.06475},
  year={2025}
}
```

  







