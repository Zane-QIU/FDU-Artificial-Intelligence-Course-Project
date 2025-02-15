import torch
from sklearn import metrics
from tqdm import tqdm
from itertools import chain

class Runner:
    def __init__(self, model, optimizer, tag_num, device="mps"):
        self.model = model
        self.optimizer = optimizer
        self.tag_num = tag_num
        self.best_score = 0.0
        self.device = device
        self.model.to(self.device)

    def train(
        self,
        train_dataloader,
        valid_dataloader,
        epochs,
        save_path,
        eval_steps,
    ):
        global_step = 0
        num_training_steps = epochs * len(train_dataloader)
        for epoch in range(1, epochs + 1):
            self.model.train()
            for text, label, seq_len in train_dataloader:
                text, label, seq_len = text.to(self.device), label.to(self.device), seq_len.to(self.device)
                loss = self.model(text, seq_len, label, mode="train")
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                print(
                    f"epoch: [{epoch}/{epochs}], "
                    + f"loss: {loss.item():2.4f}, "
                    + f"step: [{global_step}/{num_training_steps}]"
                )
                global_step += 1

                if global_step % eval_steps == 0 or global_step == num_training_steps - 1:
                    score = self.evaluate(valid_dataloader)
                    if score > self.best_score:
                        print(f"best score increase:{self.best_score} -> {score}")
                        self.best_score = score
                        self.save_model(save_path)
        
        print(f"training done best score: {self.best_score}")


    @torch.no_grad()
    def evaluate(self, valid_loader):
        self.model.eval()
        my_tags = []
        real_tags = []
        for sentence, valid_tags, sentence_len in tqdm(valid_loader):
            sentence, sentence_len = sentence.to(self.device), sentence_len.to(self.device)
            now_tags = self.model(sentence, sentence_len, mode="eval")
            for tags in now_tags:
                my_tags += tags
            for tags, now_len in zip(valid_tags, sentence_len):
                real_tags += tags[:now_len].tolist()
        score = metrics.f1_score(
            y_true=real_tags,
            y_pred=my_tags,
            labels=range(1, self.tag_num - 2),  # 排除 "O", "<START>", "<STOP>"
            average="micro",
        )
        return score

    @torch.no_grad()
    def predict(self, x):
        self.model.eval()
        logits = self.model(x, mode="pred")
        return logits

    def save_model(self, save_path):
        torch.save(self.model.state_dict(), save_path)

    def load_model(self, model_path):
        model_state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(model_state_dict)
