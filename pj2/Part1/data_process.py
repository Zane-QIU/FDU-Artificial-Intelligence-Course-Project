import os
import sys
import re
import numpy as np
import logging


def build_vocab(file_paths):
    vocab = {"UNK": 0}
    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:
                    word, _ = line.split()
                    if word not in vocab:
                        vocab[word] = len(vocab)
    return vocab


def build_tag2idx(filename):
    with open(filename, "r", encoding="utf-8") as file:
        content = file.read()
    tags = re.findall(r"[B|M|E|S|I]-[A-Z]+", content)

    if filename == "../NER/Chinese/tag.txt":
        tags = tags[4:]
    else:
        tags = tags[2:]

    tags.append("O")
    tag2idx = {tag: idx for idx, tag in enumerate(["O"] + tags[:-1], start=0)}
    return tag2idx


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

    if mode == "test":
        test_sentences, test_labels = load_from_npz(test_file)

    logging.info(f"Train dataset size: {len(train_sentences)}")
    logging.info(f"Validation dataset size: {len(valid_sentences)}")
    if mode == "test":
        logging.info(f"Test dataset size: {len(test_sentences)}")

    train_data = list(zip(train_sentences, train_labels))
    valid_data = list(zip(valid_sentences, valid_labels))
    test_data = list(zip(test_sentences, test_labels)) if mode == "test" else None

    return train_data, valid_data, test_data


def set_log(file_path):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    log_handlers = [logging.StreamHandler(stream=sys.stdout)]
    if file_path:
        log_handlers.append(logging.FileHandler(file_path))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s P%(process)d %(levelname)s %(message)s",
        handlers=log_handlers,
    )


def combine_data(sentences, labels):
    combined_data = [
        "".join(f"{token} {tag}\n" for token, tag in zip(sentence, label))
        for sentence, label in zip(sentences, labels)
    ]
    return "\n".join(combined_data) + "\n"