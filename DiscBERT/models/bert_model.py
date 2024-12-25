import torch.nn as nn
from transformers import BertModel, BertTokenizer

class hall_labeler(nn.Module):
    def __init__(self, p,args):
        super().__init__()

        self.p = p

        self.bert = BertModel.from_pretrained(args.BertModel)
        self.tokenizer = BertTokenizer.from_pretrained(args.BertModel)

        # Disable unnecessary "BertPooler" layer
        hidden_size = self.bert.pooler.dense.in_features
        self.bert.pooler = None

        # Prepare heads
        self.dropout = nn.Dropout(p)
        self.ps_heads = nn.Linear(hidden_size, 2, bias=True)
        self.pp_heads = nn.Linear(hidden_size, 2, bias=True)
        self.view_heads = nn.Linear(hidden_size, 2, bias=True)
        self.comm_heads = nn.Linear(hidden_size, 2, bias=True)

    def forward(self, report):
        device = next(self.bert.parameters()).device
        tokenized_inputs = self.tokenizer(
            report,
            padding=True,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
            pad_to_multiple_of=8)
        source_padded = tokenized_inputs.input_ids.to(device)
        attention_mask = tokenized_inputs.attention_mask.to(device)
        final_hidden = self.bert(source_padded, attention_mask=attention_mask)[0]
        cls_hidden = final_hidden[:, 0, :].squeeze(dim=1)
        cls_hidden = self.dropout(cls_hidden)

        ps_predict = self.ps_heads(cls_hidden)
        pp_predict = self.pp_heads(cls_hidden)
        view_predict = self.view_heads(cls_hidden)
        comm_predict = self.comm_heads(cls_hidden)

        return ps_predict,pp_predict,view_predict,comm_predict

    def get_tokenizer(self):
        return self.tokenizer
