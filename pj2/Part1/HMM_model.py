import numpy as np
from tqdm import tqdm

class HMM_model:
    def __init__(self, tag2id, vocab=None):
        self.tag2id = tag2id
        self.n_tag = len(self.tag2id)
        self.vocab = vocab or {}
        self.epsilon = 1e-100
        self.idx2tag = {idx: tag for tag, idx in self.tag2id.items()}
        self.A = np.zeros((self.n_tag, self.n_tag))
        self.B = np.zeros((self.n_tag, len(vocab))) if vocab else None
        self.Pi = np.zeros(self.n_tag)

    def train(self, train_data):
        for sentence, labels in tqdm(train_data):
            for j, (cur_word, cur_tag) in enumerate(zip(sentence, labels)):
                self.B[self.tag2id[cur_tag]][self.vocab[cur_word]] += 1
                if j == 0:
                    self.Pi[self.tag2id[cur_tag]] += 1
                else:
                    pre_tag = labels[j - 1]
                    self.A[self.tag2id[pre_tag]][self.tag2id[cur_tag]] += 1

        self._normalize_and_log(self.Pi, is_1d=True)
        self._normalize_and_log(self.A)
        self._normalize_and_log(self.B)
        self._set_unknown_token_probability()

    def _normalize_and_log(self, array, is_1d=False):
        if is_1d:
            array_sum = array.sum()
            if array_sum == 0:
                array_sum = self.epsilon
            array /= array_sum
            array[array == 0] = self.epsilon
            np.log10(array, out=array)
        else:
            row_sums = array.sum(axis=1)
            zero_rows = row_sums == 0
            array[~zero_rows] /= row_sums[~zero_rows, None]
            array[zero_rows] = 0
            array[array == 0] = self.epsilon
            np.log10(array, out=array)

    def _set_unknown_token_probability(self):
        self.B[:, 0] = np.log(1.0 / self.n_tag)

    def valid(self, valid_data):
        return [self.viterbi(sentence) for sentence, _ in valid_data]

    def viterbi(self, observation):
        N, T = len(self.Pi), len(observation)
        delta = np.zeros((N, T))
        psi = np.zeros((N, T), dtype=int)

        delta[:, 0] = self.Pi + self.B[:, self.vocab.get(observation[0], 0)]
        for t in range(1, T):
            O_t = self.vocab.get(observation[t], 0)
            for j in range(N):
                delta[j, t] = np.max(delta[:, t - 1] + self.A[:, j]) + self.B[j, O_t]
                psi[j, t] = np.argmax(delta[:, t - 1] + self.A[:, j])

        best_path = np.zeros(T, dtype=int)
        best_path[-1] = np.argmax(delta[:, -1])
        for t in range(T - 2, -1, -1):
            best_path[t] = psi[best_path[t + 1], t + 1]

        return [self.idx2tag[id] for id in best_path]

    def save_param(self, filename, format="txt"):
        if format == "txt":
            np.savetxt(f"{filename}_A.txt", self.A)
            np.savetxt(f"{filename}_B.txt", self.B)
            np.savetxt(f"{filename}_Pi.txt", self.Pi)
        elif format == "npy":
            np.save(f"{filename}_A.npy", self.A)
            np.save(f"{filename}_B.npy", self.B)
            np.save(f"{filename}_Pi.npy", self.Pi)
        elif format == "npz":
            np.savez(f"{filename}.npz", A=self.A, B=self.B, Pi=self.Pi)

    def load_param(self, filename, format="txt"):
        if format == "txt":
            self.A = np.loadtxt(f"{filename}_A.txt")
            self.B = np.loadtxt(f"{filename}_B.txt")
            self.Pi = np.loadtxt(f"{filename}_Pi.txt")
        elif format == "npy":
            self.A = np.load(f"{filename}_A.npy")
            self.B = np.load(f"{filename}_B.npy")
            self.Pi = np.load(f"{filename}_Pi.npy")
        elif format == "npz":
            loaded = np.load(f"{filename}.npz")
            self.A = loaded["A"]
            self.B = loaded["B"]
            self.Pi = loaded["Pi"]

    def predict(self, observation):
        results = self.viterbi(observation)
        for word, label in zip(observation, results):
            print(f"{word} {label}")