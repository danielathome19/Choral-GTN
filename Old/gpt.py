import random
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
import os
import torch
import numpy as np
import pickle as pkl
from tqdm import  tqdm
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import DataLoader, Dataset, TensorDataset
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb=10'


torch.cuda.empty_cache()


def train_choralgpt():
    # Load dataset
    if not os.path.exists("Data/Glob/Choral/Choral_vocab.pkl"):
        with open("Data/Glob/Choral/Choral.pkl", "rb") as file:
            data = pkl.load(file)

        # Iterate through the data; change all voices to numbers and START and END to -1 and -2
        voice_mapping = {"Soprano": 0, "Alto": 1, "Tenor": 2, "Bass": 3}
        for i in range(len(data)):
            for j in range(len(data[i])):
                if data[i][j][0] in voice_mapping:
                    data[i][j][0] = voice_mapping[data[i][j][0]]
                if data[i][j][1] in ["START", "END"]:
                    data[i][j][1] = -1 if data[i][j][1] == "START" else -2
                for x in range(len(data[i][j])):
                    data[i][j][x] = float(data[i][j][x]) if data[i][j][x] is not None else 0.0

        # Pad all sequences (data[i]) to the same length
        data = pad_sequences(data, padding="post", dtype=np.float32)
        vocab = np.array(data, dtype=np.float32)

        # Iterate through all sequences (data[i]); split them into chunks of 1024 tokens


        def chunk_midi(midi, chunk_size=1024):
            """Splits a MIDI's tokens into chunks of a specified size."""
            num_chunks = len(midi) // chunk_size
            chunks = [midi[i * chunk_size: (i + 1) * chunk_size] for i in range(num_chunks)]

            # Handle the last chunk
            if len(midi) % chunk_size:
                last_chunk = midi[num_chunks * chunk_size:]
                # If you want to pad it:
                padding = np.zeros((chunk_size - len(last_chunk), midi.shape[1]))
                last_chunk_padded = np.concatenate([last_chunk, padding])
                chunks.append(last_chunk_padded)

            return chunks


        chunked_data = []
        for midi in data:
            chunked_data.extend(chunk_midi(midi))

        chunked_data = np.array(chunked_data)
        vocab = chunked_data
    else:
        with open("Data/Glob/Choral/Choral_vocab.pkl", "rb") as file:
            vocab = pkl.load(file)
        chunked_data = vocab
    # print(chunked_data.shape)

    scaler = MinMaxScaler(feature_range=(0, 1))  # Normalization
    if os.path.exists("Weights/Composition/Choral_scaler.pkl"):
        with open(f"Weights/Composition/Choral_scaler.pkl", "rb") as f:
            scaler = pkl.load(f)
            scaled_data = scaler.transform(np.array(vocab).reshape(-1, 1))
    else:
        scaled_data = scaler.fit_transform(np.array(vocab).reshape(-1, 1))
    vocab = scaled_data.reshape(chunked_data.shape)
    # Get the vocab size from each token (data[i][j]) as the max value in each token
    vocab_size = [int(np.max(vocab[:, :, i])) + 1 for i in range(vocab.shape[-1])]
    # vocab_size = int(np.max(vocab)) + 1

    # Save vocabulary and scaler if the files don't exist
    if not os.path.exists("Weights/Composition/Choral_vocab.pkl"):
        with open(f"Weights/Composition/Choral_vocab.pkl", "wb") as f:
            pkl.dump(vocab, f)
    if not os.path.exists("Weights/Composition/Choral_scaler.pkl"):
        with open(f"Weights/Composition/Choral_scaler.pkl", "wb") as f:
            pkl.dump(scaler, f)

    # Assuming data has been converted to integer tokens and split into inputs and targets
    inputs = torch.tensor(vocab[:-1], dtype=torch.long)
    targets = torch.tensor(vocab[1:], dtype=torch.long)

    class MIDIDataset(Dataset):
        def __init__(self, inputs, targets):
            self.inputs = inputs
            self.targets = targets

        def __len__(self):
            return len(self.inputs)

        def __getitem__(self, idx):
            return self.inputs[idx], self.targets[idx]

    # Hyperparameters
    BATCH_SIZE = 1
    EPOCHS = 3
    LEARNING_RATE = 3e-4
    WARMUP_STEPS = 100

    # Create Dataloaders
    dataset = MIDIDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Load pre-trained model and tokenizer
    load_from_checkpoint = False
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.resize_token_embeddings(len(vocab))  # Assuming `vocab` is the token-to-int mapping
    if load_from_checkpoint:
        model_state_dict = torch.load("Weights/ChoralGPT/ChoralGPT_1.pt")
        model.load_state_dict(model_state_dict)

    # Move model to GPU if available
    print("CUDA available:", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    model.to(device)

    # Optimizer & Learning Rate Scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS,
                                                num_training_steps=len(dataloader) * EPOCHS)

    # Training Loop
    print("Starting training")
    # for epoch in range(EPOCHS):
    #     # Wrap dataloader with tqdm for progress bar
    #     pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
    #     for batch_inputs, batch_targets in pbar:
    #         batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
    #         outputs = model(batch_inputs, labels=batch_targets)
    #         loss = outputs.loss
    #
    #         optimizer.zero_grad()
    #         loss.backward()
    #         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    #         optimizer.step()
    #         scheduler.step()
    #
    #         # Update tqdm progress bar with loss
    #         pbar.set_postfix({'Loss': loss.item()})
    #
    #     print(f"Epoch {epoch + 1}/{EPOCHS} - Loss: {loss.item()}")

    GRAD_ACCUM_STEPS = 2  # Number of batches over which to accumulate gradient.
    # This effectively multiplies the batch size by this factor.
    print(f"Effective Batch Size: {BATCH_SIZE * GRAD_ACCUM_STEPS}")

    for epoch in range(EPOCHS):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}", total=len(dataloader))
        model.zero_grad()  # Initialize gradients once at the start of the epoch

        for step, (batch_inputs, batch_targets) in enumerate(pbar):  # Use enumerate over pbar
            batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
            outputs = model(batch_inputs, labels=batch_targets)
            loss = outputs.loss

            # This scales loss. Otherwise, the loss will be the average of all batches.
            loss = loss / GRAD_ACCUM_STEPS
            loss.backward()

            if (step + 1) % GRAD_ACCUM_STEPS == 0 or step == len(dataloader) - 1:  # step + 1 because step starts at 0
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                model.zero_grad()  # Important to clear the accumulated gradients

            pbar.set_postfix({'Loss': loss.item()})  # Display current loss in the progress bar

        print(f"Epoch {epoch + 1}/{EPOCHS} - Loss: {loss.item()}")
        torch.save(model.state_dict(), f"Weights/ChoralGPT/ChoralGPT_{epoch + 1}.pt")
        # Epoch 1/3: 100% 19959/19959 [2:00:17<00:00,  2.77it/s, Loss=0]

    # Save model
    model.save_pretrained("Weights/ChoralGPT")


def generate_choralgpt():
    # Load scaler
    with open("Weights/Composition/Choral_scaler.pkl", "rb") as f:
        scaler = pkl.load(f)

    # Load vocabulary
    with open("Weights/Composition/Choral_vocab.pkl", "rb") as f:
        vocab = pkl.load(f)

    # Load model from checkpoint
    print("CUDA available:", torch.cuda.is_available())
    # device = torch.device("cpu")
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.resize_token_embeddings(len(vocab))
    model_state_dict = torch.load("Weights/ChoralGPT/ChoralGPT_1.pt", map_location=device)
    model.load_state_dict(model_state_dict)

    # Move model to GPU if available
    model.to(device)

    model.eval()

    for i in range(1):
        seed = torch.tensor([[random.randint(0, len(vocab))]], dtype=torch.long).to(device)
        output = model.generate(seed, max_length=1024, do_sample=True, top_k=100, top_p=0.9, temperature=1.0)
        output = output.cpu().numpy().squeeze()  # Remove batch dimension

        # Map the token IDs back to their scaled 5-dimensional representation
        scaled_representation = [vocab[token_id][0] for token_id in output]  # Using [0] to access the inner list

        # Convert the list of lists into a 2D numpy array
        scaled_representation = np.array(scaled_representation)

        # Use the inverse_transform to get the original values
        output_values = scaler.inverse_transform(scaled_representation)
        for token in output_values:
            print(token)
        # print(output_values)

    pass


if __name__ == "__main__":
    print("Hello, world!")
    # train_choral_gpt()
    generate_choralgpt()
