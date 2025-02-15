import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


def argmax(vec):
    _, idx = torch.max(vec, 1)
    return idx.item()


def log_sum_exp(vec):
    max_score, _ = torch.max(vec, dim=-1)
    max_score_broadcast = max_score.unsqueeze(-1).repeat_interleave(vec.shape[-1], dim=-1)
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast), dim=-1))


class TransformerCRF(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab, label_map, device="cpu"):
        super(TransformerCRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = len(vocab)
        self.tagset_size = len(label_map)
        self.device = device

        self.word_embeds = nn.Embedding(self.vocab_size, embedding_dim)
        self.dropout = nn.Dropout(p=0.5)

        self.encoder_layer = TransformerEncoderLayer(d_model=embedding_dim, nhead=8)
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers=2)

        self.hidden2tag = nn.Linear(embedding_dim, self.tagset_size, bias=True)
        self.crf = CRF(label_map, device)

    def _get_transformer_features(self, sentence, seq_len):
        embeds = self.word_embeds(sentence)
        embeds = self.dropout(embeds)
        transformer_out = self.transformer_encoder(embeds)
        transformer_feats = self.hidden2tag(transformer_out)
        return transformer_feats

    def forward(self, sentence, seq_len, tags=None, mode="train"):
        feats = self._get_transformer_features(sentence, seq_len)
        if mode == "train":
            if tags is None:
                raise ValueError("In training mode, tags must be provided")
            loss = self.crf.neg_log_likelihood(feats, tags, seq_len)
            return loss
        elif mode == "eval":
            all_tag = []
            for i, feat in enumerate(feats):
                all_tag.append(self.crf._viterbi_decode(feat[:seq_len[i]])[1])
            return all_tag
        elif mode == "pred":
            return self.crf._viterbi_decode(feats[0])[1]


class CRF(nn.Module):
    def __init__(self, label_map, device="cpu"):
        super(CRF, self).__init__()
        self.label_map = label_map
        self.label_map_inv = {v: k for k, v in label_map.items()}
        self.tagset_size = len(self.label_map)
        self.device = device

        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size).to(self.device))
        self.START_TAG = "<START>"
        self.STOP_TAG = "<STOP>"
        self.transitions.data[self.label_map[self.START_TAG], :] = -10000
        self.transitions.data[:, self.label_map[self.STOP_TAG]] = -10000

    def _forward_alg(self, feats, seq_len):
        init_alphas = torch.full((self.tagset_size,), -10000.0, device=self.device)
        init_alphas[self.label_map[self.START_TAG]] = 0.0

        forward_var = torch.full((feats.shape[0], feats.shape[1] + 1, feats.shape[2]), -10000.0, device=self.device)
        forward_var[:, 0, :] = init_alphas

        for seq_i in range(feats.shape[1]):
            emit_score = feats[:, seq_i, :]
            tag_var = forward_var[:, seq_i, :].unsqueeze(1) + self.transitions + emit_score.unsqueeze(2)
            forward_var[:, seq_i + 1, :] = log_sum_exp(tag_var)

        forward_var = forward_var[range(feats.shape[0]), seq_len, :]
        terminal_var = forward_var + self.transitions[self.label_map[self.STOP_TAG]].unsqueeze(0).repeat(feats.shape[0], 1)
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _score_sentence(self, feats, tags, seq_len):
        score = torch.zeros(feats.shape[0], device=self.device)
        start = torch.tensor([self.label_map[self.START_TAG]], device=self.device).unsqueeze(0).repeat(feats.shape[0], 1)
        tags = torch.cat([start, tags], dim=1)
        for batch_i in range(feats.shape[0]):
            score[batch_i] = (
                torch.sum(self.transitions[tags[batch_i, 1:seq_len[batch_i] + 1], tags[batch_i, :seq_len[batch_i]]])
                + torch.sum(feats[batch_i, range(seq_len[batch_i]), tags[batch_i][1:seq_len[batch_i] + 1]])
            )
            score[batch_i] += self.transitions[self.label_map[self.STOP_TAG], tags[batch_i][seq_len[batch_i]]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        init_vvars = torch.full((1, self.tagset_size), -10000.0, device=self.device)
        init_vvars[0][self.label_map[self.START_TAG]] = 0

        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []
            viterbivars_t = []

            for next_tag in range(self.tagset_size):
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        terminal_var = forward_var + self.transitions[self.label_map[self.STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)

        start = best_path.pop()
        assert start == self.label_map[self.START_TAG]
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, feats, tags, seq_len):
        forward_score = self._forward_alg(feats, seq_len)
        gold_score = self._score_sentence(feats, tags, seq_len)
        return torch.mean(forward_score - gold_score)