import torch
from torch.utils.data import Dataset
from transformers import LEDTokenizer

class CNFDataset(Dataset):
    def __init__(self, source_data, tokenizer):
        self.source_data = [line.strip() for line in source_data]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.source_data)

    def __getitem__(self, index):
        src_text = self.source_data[index]
        src_encoded = self.tokenizer.encode_plus(src_text, add_special_tokens=True, return_tensors="pt", padding=False, truncation=True)
        return src_encoded['input_ids'].squeeze(0), src_encoded['attention_mask'].squeeze(0)



def load_data(source_file):
    with open(source_file, 'r') as src_file:
        source_data = src_file.readlines()
    return source_data


def collate_fn(batch):
    src_ids, attention_masks = zip(*batch)

    # Use tokenizer's pad token ID for padding
    pad_token_id = LEDTokenizer.from_pretrained('allenai/led-base-16384').pad_token_id

    # Padding sequences dynamically to the max length in each batch
    src_ids_padded = torch.nn.utils.rnn.pad_sequence(src_ids, batch_first=True, padding_value=pad_token_id)
    attention_masks_padded = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0)

    return src_ids_padded, attention_masks_padded

