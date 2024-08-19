import argparse
import torch
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
from transformers import LEDTokenizer, LEDForConditionalGeneration
from torch.utils.data import DataLoader
from data_utils_gen_pred import CNFDataset, collate_fn

def load_source_data(source_file):
    with open(source_file, 'r') as src_file:
        return [line.strip() for line in src_file.readlines()]

def main(source_file,model_path,output_path):
    print('Script Started')


 #   device = "cpu"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print('Script Started')
    # Load source data
    source_data = load_source_data(source_file)
    print('Test data loaded')
    # Initialize tokenizer and model
    tokenizer = LEDTokenizer.from_pretrained('allenai/led-base-16384')
    model = LEDForConditionalGeneration.from_pretrained('allenai/led-base-16384')
    state_dict = torch.load(model_path, map_location = device)['model_state_dict']
    adjusted_state_dict = {k[len("module."):]: v for k, v in state_dict.items() if k.startswith("module.")}
    # Load the adjusted state dict
    model.load_state_dict(adjusted_state_dict)
    
    
 #   device = 'cpu'
    model.to(device)
    model.eval()

    # Create data loader for the source data
    test_dataset = CNFDataset(source_data, tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=10, shuffle=False, collate_fn=collate_fn)
    print('Created data loader')
    for batch in test_dataloader:
        src_ids, attention_masks = batch[0], batch[1]
        
        # Decode the source ids to text for the first batch
        for idx in range(src_ids.size(0)):  # Iterate over the batch size
            decoded_text = tokenizer.decode(src_ids[idx], skip_special_tokens=True)

        break
    # Predict and format CNF expressions
    predictions = []
    print('Generating predictions')
    with torch.no_grad():
        for batch in test_dataloader:
            # Move input tensors to the same device as the model
            src_ids, attention_masks = batch[0].to(device), batch[1].to(device)

            # Generate predictions
            outputs = model.generate(input_ids=src_ids, attention_mask=attention_masks, max_length=2000)
            decoded_predictions = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outputs]
            predictions.extend(decoded_predictions)
    # Save predictions to a file
    with open(output_path, "w") as file:
        for pred in predictions:
            file.write(f"{pred}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate CNF formula predictions')
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    args = parser.parse_args()
    main(args.input,args.model_path,args.output)
