class DaTuDataset(Dataset):
    def __init__(self, sequences, scores, tokenizer, max_length=512):
        self.sequences = sequences
        self.scores = scores
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        score = self.scores[idx]
        
        # Tokenize the sequence
        inputs = self.tokenizer.encode_plus(
            sequence,
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=self.max_length,  # Max length to truncate/pad
            padding='max_length',  # Pad to max_length
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'  # Return PyTorch tensors
        )
        return {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'labels': torch.tensor(score, dtype=torch.float)
        }
        #inputs.input_ids.squeeze(0), inputs.attention_mask.squeeze(0), torch.tensor(score, dtype=torch.float)

# Training loop
def train_val_loop(model, train_dataloader, val_dataloader, optimizer, scheduler, epochs):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        i = 0
        for batch in train_dataloader:

            batch = {k: v.squeeze(1).to(device) for k, v in batch.items()}  # Move batch to device
            outputs = model(**batch)

            # input_ids, attention_mask, labels = [b.to(device) for b in batch]
            # outputs = model(input_ids, attention_mask=attention_mask, labels=labels.float())
            
            loss = outputs.loss
            total_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()
            scheduler.step()
            
        avg_train_loss = total_loss / len(train_dataloader)
        wandb.log({"train_loss": avg_train_loss})

        # Validation loop
        model.eval()
        val_loss = 0
        for batch in val_dataloader:
            i+=1
            batch = {k: v.squeeze(1).to(device) for k, v in batch.items()}  # Move batch to device
            with torch.no_grad():
                outputs = model(**batch)
                loss = outputs.loss
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_dataloader)
        wandb.log({"validation_loss": avg_val_loss})

        print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.3f} | Validation Loss: {avg_val_loss:.3f}")

class CurriculumDaTuDatasett(Dataset):
    def __init__(self, sequences, scores, tokenizer, max_length=512):
        self.sequences = sequences
        self.scores = scores
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Compute complexity of each sequence (e.g., length of the sequence)
        self.complexity = [len(seq) for seq in sequences]

        # Sort the dataset by complexity
        self._sort_by_complexity()

    def _sort_by_complexity(self):
        # Sort sequences, problem_ids, grades, and complexity based on complexity
        combined = sorted(zip(self.sequences, self.scores, self.complexity), key=lambda x: x[2])
        self.sequences, self.scores, self.complexity = zip(*combined)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        score = self.scores[idx]

        # Tokenize the sequence
        inputs = self.tokenizer.encode_plus(
            sequence,
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=self.max_length,  # Max length to truncate/pad
            padding='max_length',  # Pad to max_length
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'  # Return PyTorch tensors
        )
        return {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'labels': torch.tensor(score, dtype=torch.float)
        }