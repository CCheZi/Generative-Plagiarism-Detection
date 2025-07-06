import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
import spacy
import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from tqdm import tqdm

nlp = spacy.load('en_core_web_sm')

def read_and_split(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    doc = nlp(text)
    sentences = []
    offsets = []
    for sent in doc.sents:
        start = sent.start_char
        end = sent.end_char
        sentences.append(sent.text.strip())
        offsets.append((start, end - start))
    return sentences, offsets, text

def vectorize_sentences(sentences, model):
    embeddings = model.encode(sentences, convert_to_numpy=True)
    return embeddings

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index

def bert_predict(model, tokenizer, sentence1, sentence2):
    inputs = tokenizer(sentence1, sentence2, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        return probs[0][1].item()

def merge_blocks(matched_pairs, max_susp_gap=5, max_src_gap=5, min_block_size=2):
    if not matched_pairs:
        return []
    matched_pairs.sort()
    merged = []
    current_block = [matched_pairs[0]]
    for i in range(1, len(matched_pairs)):
        prev_susp, prev_src = current_block[-1]
        curr_susp, curr_src = matched_pairs[i]
        if (
            (curr_susp - prev_susp <= max_susp_gap)
            and (abs(curr_src - prev_src) <= max_src_gap)
            and (curr_susp > prev_susp and curr_src > prev_src)
        ):
            current_block.append((curr_susp, curr_src))
        else:
            if len(current_block) >= min_block_size:
                merged.append(current_block)
            current_block = [matched_pairs[i]]
    if len(current_block) >= min_block_size:
        merged.append(current_block)
    return merged

def write_xml(output_path, susp_name, src_name, merged_blocks, susp_offsets, src_offsets, susp_body_offset=0, src_body_offset=0):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    doc = ET.Element("document", reference=susp_name)
    for block_group in merged_blocks:
        susp_start = susp_offsets[block_group[0][0]][0] + susp_body_offset
        susp_end = susp_offsets[block_group[-1][0]][0] + susp_offsets[block_group[-1][0]][1] + susp_body_offset
        src_start = src_offsets[block_group[0][1]][0] + src_body_offset
        src_end = src_offsets[block_group[-1][1]][0] + src_offsets[block_group[-1][1]][1] + src_body_offset
        ET.SubElement(doc, "feature", {
            "name": "detected-plagiarism",
            "this_offset": str(susp_start),
            "this_length": str(susp_end - susp_start),
            "source_reference": src_name,
            "source_offset": str(src_start),
            "source_length": str(src_end - src_start)
        })
    xml_str = minidom.parseString(ET.tostring(doc)).toprettyxml(indent="  ")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(xml_str)
    print(f"已生成 XML 文件: {output_path}")

def process_pair(susp_path, src_path, output_path, sbert_model, mpnet_model, bert_model, bert_tokenizer):
    print(f"\n开始处理文件对:\n  抄袭文件: {susp_path}\n  源文件: {src_path}")
    susp_sentences, susp_offsets, _ = read_and_split(susp_path)
    src_sentences, src_offsets, _ = read_and_split(src_path)
    print(f"分句完成：抄袭文档句子数={len(susp_sentences)}, 源文档句子数={len(src_sentences)}")

    susp_embeddings = vectorize_sentences(susp_sentences, sbert_model)
    src_embeddings = vectorize_sentences(src_sentences, sbert_model)
    mpnet_susp_embeddings = vectorize_sentences(susp_sentences, mpnet_model)
    mpnet_src_embeddings = vectorize_sentences(src_sentences, mpnet_model)

    index = build_faiss_index(src_embeddings)
    mpnet_index = build_faiss_index(mpnet_src_embeddings)

    vectorizer = TfidfVectorizer(ngram_range=(1,2)).fit(src_sentences + susp_sentences)
    src_tfidf = vectorizer.transform(src_sentences)
    susp_tfidf = vectorizer.transform(susp_sentences)

    matched_pairs = []
    for i, (susp_vec, susp_embed, mpnet_embed) in enumerate(tqdm(zip(susp_tfidf, susp_embeddings, mpnet_susp_embeddings), total=len(susp_sentences), ncols=80)):
        query = susp_embed.reshape(1, -1)
        faiss.normalize_L2(query)
        D, I = index.search(query, 5)
        faiss_indices = I[0]
        faiss_scores = D[0]

        tfidf_sim = cosine_similarity(susp_vec, src_tfidf).flatten()
        tfidf_indices = np.argsort(tfidf_sim)[-5:]

        mpnet_query = mpnet_embed.reshape(1, -1)
        faiss.normalize_L2(mpnet_query)
        D2, I2 = mpnet_index.search(mpnet_query, 5)
        mpnet_indices = I2[0]
        mpnet_scores = D2[0]

        candidates = set(faiss_indices).union(tfidf_indices).union(mpnet_indices)

        if faiss_scores[0] > 0.35:
            matched_pairs.append((i, faiss_indices[0]))
            continue

        if tfidf_sim[tfidf_indices[-1]] > 0.55:
            matched_pairs.append((i, tfidf_indices[-1]))
            continue

        if mpnet_scores[0] > 0.5:
            matched_pairs.append((i, mpnet_indices[0]))
            continue

        for idx in candidates:
            score = bert_predict(bert_model, bert_tokenizer, susp_sentences[i], src_sentences[idx])
            if score > 0.45:
                matched_pairs.append((i, idx))
                break

    merged_blocks = merge_blocks(matched_pairs)
    write_xml(output_path, os.path.basename(susp_path), os.path.basename(src_path), merged_blocks, susp_offsets, src_offsets)

if __name__ == '__main__':  
    print("加载模型...")
    sbert_model = SentenceTransformer('models/all-MiniLM-L6-v2')
    mpnet_model = SentenceTransformer('models/all-mpnet-base-v2')
    bert_tokenizer = BertTokenizer.from_pretrained('models/bert_base_uncased_finetuned')
    bert_model = BertForSequenceClassification.from_pretrained('models/bert_base_uncased_finetuned')
    print("模型加载完成")

    input_dir = os.environ.get("inputDataset", "input")    # 用于本地调试
    output_dir = os.environ.get("outputDir", "output")

    pairs_file = os.path.join(input_dir, "pairs", "pairs")
    sups_dir = os.path.join(input_dir, "sups")
    src_dir = os.path.join(input_dir, "src")

    with open(pairs_file, 'r', encoding='utf-8') as f:
        pairs = [line.strip().split() for line in f if line.strip()]

    for susp_file, src_file in pairs:
        susp_path = os.path.join(sups_dir, susp_file)
        src_path = os.path.join(src_dir, src_file)
        output_file = f"{os.path.splitext(susp_file)[0]}-{os.path.splitext(src_file)[0]}.xml"
        output_path = os.path.join(output_dir, output_file)
        process_pair(susp_path, src_path, output_path, sbert_model, mpnet_model, bert_model, bert_tokenizer)

    print("全部处理完成！")

