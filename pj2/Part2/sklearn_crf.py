import re
import os
import sys
import numpy as np
import logging

def build_vocab(file_paths):
    vocab = {"UNK": 0}
    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:
                    word, _ = line.split()  # 分割每行的单词和标签
                    if word not in vocab:
                        vocab[word] = len(vocab)
    return vocab

def build_tag2idx(filename):
    with open(filename, "r", encoding="utf-8") as f:
        content = f.read()
        tags = re.findall(r"[B|M|E|S|I]-[A-Z]+", content)
        
    if filename == "../NER/Chinese/tag.txt":
        tags = tags[4:]
    else:
        tags = tags[2:]
    
    tags.append("O")
    tag2idx = {tag: idx for idx, tag in enumerate(["O"] + tags)}
    return tag2idx

def read_data(file_path):
    sentences, labels = [], []
    curr_sentence, curr_labels = [], []

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line == "":
                if curr_sentence:
                    sentences.append(curr_sentence)
                    labels.append(curr_labels)
                    curr_sentence, curr_labels = [], []
            else:
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
        logging.info(f"test dataset size: {len(test_sentences)}")
        test_data = list(zip(test_sentences, test_labels))
    else:
        test_data = None

    logging.info(f"train dataset size: {len(train_sentences)}")
    logging.info(f"valid dataset size: {len(valid_sentences)}")

    train_data = list(zip(train_sentences, train_labels))
    valid_data = list(zip(valid_sentences, valid_labels))
    
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

def word2features_0(sent, i):
    word = sent[i]
    prev_word = "<start>" if i == 0 else sent[i - 1]
    next_word = "<end>" if i == (len(sent) - 1) else sent[i + 1]
    prev_word2 = "<start>" if i <= 1 else sent[i - 2]
    next_word2 = "<end>" if i >= (len(sent) - 2) else sent[i + 2]

    features = {
        "w": word,
        "w-1": prev_word,
        "w+1": next_word,
        "w-1:w": prev_word + word,
        "w:w+1": word + next_word,
        "w-1:w:w+1": prev_word + word + next_word,
        "w-2:w": prev_word2 + word,
        "w:w+2": word + next_word2,
        "bias": 1,
        "word.isdigit": word.isdigit(),
    }
    return features

def en2features(sent, i):
    word = sent[i]
    features = {
        "bias": 1.0,
        "word.lower()": word.lower(),
        "word[-3:]": word[-3:],
        "word[-2:]": word[-2:],
        "word[:3]": word[:3],
        "word[:2]": word[:2],
        "word.isupper()": word.isupper(),
        "word.istitle()": word.istitle(),
        "word.isdigit()": word.isdigit(),
        "word": word,
        "word.length()": len(word),
        "word.isalnum()": word.isalnum(),
        "word.has_hyphen()": "-" in word,
        "word.has_digit()": any(char.isdigit() for char in word),
    }

    if i > 0:
        prev_word = sent[i - 1]
        features.update({
            "prev_word.lower()": prev_word.lower(),
            "prev_word.isupper()": prev_word.isupper(),
            "prev_word[-3:]": prev_word[-3:],
            "prev_word[-2:]": prev_word[-2:],
            "prev_word[:3]": prev_word[:3],
            "prev_word[:2]": prev_word[:2],
        })
    else:
        features["BOS"] = True

    if i < len(sent) - 1:
        next_word = sent[i + 1]
        features.update({
            "next_word.lower()": next_word.lower(),
            "next_word.isupper()": next_word.isupper(),
            "next_word[-3:]": next_word[-3:],
            "next_word[-2:]": next_word[-2:],
            "next_word[:3]": next_word[:3],
            "next_word[:2]": next_word[:2],
        })
    else:
        features["EOS"] = True

    return features

def word2features_1(sent, language, i):
    if language == "English":
        return en2features(sent, i)
    elif language == "Chinese":
        return word2features_0(sent, i)

def sent2features(sent, language, feature_func_num=0):
    if feature_func_num == 0:
        feature_func = word2features_0
    elif feature_func_num == 1:
        feature_func = word2features_1
    elif feature_func_num == 2:
        raise NotImplementedError
    
    return [feature_func(sent, language, i) for i in range(len(sent))]