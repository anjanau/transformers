from torch.nn import Module, Linear
from transformers import BertModel
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss

class BertWithLinearLayer(Module):
    def __init__(self, bert_pretrained_path, num_labels=2):
        super(BertWithLinearLayer, self).__init__()
        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained(bert_pretrained_path, cache_dir=None)
        self.fc1 = Linear(768, 512)
        self.out = Linear(512, 2)
        # self.dropout = Dropout(HIDDEN_DROPOUT)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,):

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                      token_type_ids=token_type_ids)
        x = F.relu(self.fc1(outputs[1]))
        logits = self.out(x)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        return cls(pretrained_model_name_or_path)

    def save_pretrained(self, save_directory):
        pass