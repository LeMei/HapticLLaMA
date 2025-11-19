from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer
import nltk
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import re
import json
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics.pairwise import cosine_distances
from sklearn.decomposition import PCA

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = SentenceTransformer('all-MiniLM-L6-v2')  # SBERT 模型（小而快）

nltk.download('punkt_tab')

def extract_numbered_sentences(text):
    pattern = r'^\d+:(.*)$'
    matches = re.findall(pattern, text, re.MULTILINE)
    return [m.strip() for m in matches]

def extract_signal_blocks(text):
    blocks = re.split(r'-{10,}', text)
    results = []
    step = 2
    for i in range(1, len(blocks), step):
        block = blocks[i]
        meta_data = block
        match_meta = re.search(r'signal id:([^\s,]+),\s*category:([^\s]+)', meta_data)
        if match_meta:
            signal_id = match_meta.group(1).strip()
            category = match_meta.group(2).strip()
        caption_data = blocks[i+1]
        match_cap =re.search(r'^\d+:', caption_data, re.MULTILINE)
        if match_cap:
            captions = extract_numbered_sentences(caption_data)
        results.append({
            "signal_id": signal_id,
            "category": category,
            "captions": captions
        })

    return results

def visualization_signal(signals):

    K = 6 
    signal_features = model.encode(signals)

    signal_emb_pca = PCA(n_components=20).fit_transform(signal_features)
    signal_clusters = KMeans(n_clusters=K, random_state=42).fit_predict(signal_emb_pca)

def visualization_caption(captions, category, sigids):

    embeddings = model.encode(captions)

    K = 6  
    kmeans = KMeans(n_clusters=K, random_state=42)
    labels = kmeans.fit_predict(embeddings)

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings)


    plt.figure(figsize=(10, 8))
    palette = sns.color_palette("hsv", K)
    sns.scatterplot(x=embeddings_2d[:, 0], y=embeddings_2d[:, 1], hue=labels, palette=palette, s=30, linewidth=0)
    plt.title("Generated Caption Clusters (K-Means) in Semantic Space")
    plt.xlabel("TSNE Dim 1")
    plt.ylabel("TSNE Dim 2")
    plt.legend(title="Cluster", loc='best', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()

    plt.savefig('./{}_caption_cluster.png'.format(category), dpi=450)

def caption_readin(file_path):
    sensory, emotion, association = [], [], []
    sen_sid, emo_sid, ass_sid = []

    all_cap = []

    with open(file_path, "r") as f:
        raw_text = f.read()

    data = extract_signal_blocks(raw_text)

    with open("haptic_caption_data.json", "w") as out:
        json.dump(data, out, indent=2)
    for item in data:
        if item['category'] == 'sensory':
            sen_sid.append(item["signal_id"])
            for i, cap in enumerate(item["captions"]):
                sensory.append(cap)
                all_cap.append(cap)
        
        if item['category'] == 'emotion':
            emo_sid.append(item["signal_id"])
            for i, cap in enumerate(item["captions"]):
                emotion.append(cap)
                all_cap.append(cap)


        if item['category'] == 'association':
            ass_sid.append(item["signal_id"])
            for i, cap in enumerate(item["captions"]):
                association.append(cap)
                all_cap.append(cap)

    print('sen_len:{}, emo_len:{}, ass_len:{}'.format(len(sensory), len(emotion), len(association)))
    visualization_caption(sensory, 'sensory')
    visualization_caption(emotion, 'emotion')
    visualization_caption(association, 'association')
    # visualization_caption(all_cap, 'all_cap')


file_path = r'../inference_caption_epoch1_5.1.txt'
caption_readin(file_path=file_path)



    
    
        