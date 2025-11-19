import token
import torch
import yaml

from datasets import Dataset, Audio
from repcodec.RepCodec import RepCodec
from feature_extractor import Feature_Extractor


config = "../repcodec/configs/repcodec_dim768.yaml"
trained_model_path = '../exp/output_encodec_converted/checkpoint-20000steps.pkl'
with open(config) as fp:
    conf = yaml.load(fp, Loader=yaml.FullLoader)

extractor = Feature_Extractor(extractor_name='', sample_rate=16000)

model = RepCodec(**conf)
# model.load_state_dict(torch.load("./hubert_base_l9.pkl", map_location="cpu")["model"]["repcodec"])
model.load_state_dict(torch.load(trained_model_path, map_location="cpu")["model"]["repcodec"])
model.quantizer.initial()
model.eval()

haptic_set = ['../data/Haptic_manifest/signal/aug_signal/F1_loop_aug0.wav']
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
        print(tokens,len(tokens))