import os
import re
import numpy as np

sorted_labels_eng = [
    "<PAD>", "O", "B-PER", "I-PER", "B-ORG", "I-ORG", 
    "B-LOC", "I-LOC", "B-MISC", "I-MISC"
]

sorted_labels_chn = [
    "<PAD>", "O", "B-NAME", "M-NAME", "E-NAME", "S-NAME",
    "B-CONT", "M-CONT", "E-CONT", "S-CONT", "B-EDU", "M-EDU",
    "E-EDU", "S-EDU", "B-TITLE", "M-TITLE", "E-TITLE", "S-TITLE",
    "B-ORG", "M-ORG", "E-ORG", "S-ORG", "B-RACE", "M-RACE",
    "E-RACE", "S-RACE", "B-PRO", "M-PRO", "E-PRO", "S-PRO",
    "B-LOC", "M-LOC", "E-LOC", "S-LOC"
]


def read_data(file_path):
    sentences, labels = [], []
    curr_sentence, curr_labels = [], []

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                if curr_sentence:
                    sentences.append(curr_sentence)
                    labels.append(curr_labels)
                    curr_sentence, curr_labels = [], []
                continue
            token, label = line.split(" ")
            curr_sentence.append(token)
            curr_labels.append(label)

    if curr_sentence:
        sentences.append(curr_sentence)
        labels.append(curr_labels)

    return sentences, labels


def save_as_npz(sentences, labels, file_path):
    np.savez(file_path, sentences=np.array(sentences, dtype=object), labels=np.array(labels, dtype=object))


def load_from_npz(file_path):
    loaded_data = np.load(file_path, allow_pickle=True)
    return loaded_data["sentences"], loaded_data["labels"]


def data_process(folder_path, mode="train"):
    train_file = os.path.join(folder_path, "train.npz")
    valid_file = os.path.join(folder_path, "valid.npz")
    test_file = os.path.join(folder_path, "test.npz")

    if not os.path.isfile(train_file):
        train_sentences, train_labels = read_data(os.path.join(folder_path, "train.txt"))
        save_as_npz(train_sentences, train_labels, train_file)

    if not os.path.isfile(valid_file):
        valid_sentences, valid_labels = read_data(os.path.join(folder_path, "validation.txt"))
        save_as_npz(valid_sentences, valid_labels, valid_file)

    if mode == "test" and not os.path.isfile(test_file):
        test_sentences, test_labels = read_data(os.path.join(folder_path, "test.txt"))
        save_as_npz(test_sentences, test_labels, test_file)

    train_sentences, train_labels = load_from_npz(train_file)
    valid_sentences, valid_labels = load_from_npz(valid_file)

    train_data = list(zip(train_sentences, train_labels))
    valid_data = list(zip(valid_sentences, valid_labels))
    test_data = list(zip(load_from_npz(test_file))) if mode == "test" else None

    return train_data, valid_data, test_data


def build_word2id(filename):
    word2id = {"PAD": 0, "UNK": 1}
    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            word = line.strip().split()[0]
            if word not in word2id:
                word2id[word] = len(word2id)

    id2word = {v: k for k, v in word2id.items()}
    return word2id, id2word


def build_tag2id(filename):
    with open(filename, "r", encoding="utf-8") as file:
        content = file.read()

    tags = re.findall(r"[B|M|E|S|I]-[A-Z]+", content)
    tags = tags[4:] if filename == "../NER/Chinese/tag.txt" else tags[2:]
    tags.append("O")

    tag2id = {tag: idx for idx, tag in enumerate(tags)}
    tag2id["<START>"] = len(tag2id)
    tag2id["<STOP>"] = len(tag2id)

    id2tag = {v: k for k, v in tag2id.items()}
    return tag2id, id2tag