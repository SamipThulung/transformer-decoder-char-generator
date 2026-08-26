import torch
import string



def predict_names(
    model, 
    num_names=20,
    max_length=19,
    temperature=0.8
):
    TOKENS = 20
    START_TOKEN = 0
    END_TOKEN = 27
    PAD_TOKEN = 28
    
    id_to_char = {
        0: "<start>",
        **{i + 1: c for i, c in enumerate(string.ascii_lowercase)},
        27: "<end>",
        28: "<pad>",
    }

    names = []

    model.eval()

    with torch.no_grad():

        for _ in range(num_names):
            # Start every name with <start>
            sequence = [START_TOKEN]
            for _ in range(max_length):
                # Padding Token
                input_sequence = sequence + [PAD_TOKEN] * (
                    TOKENS - len(sequence)
                )

                # Make sure sequence does not exceed TOKENS
                input_sequence = input_sequence[:TOKENS]
                # Tensor shape:
                # [1, 20]
                x = torch.tensor(
                    input_sequence,
                    dtype=torch.long
                ).unsqueeze(0)
                logits = model(x)

                logits = logits[:, -1, :]

                logits = logits / temperature

                probabilities = torch.softmax(
                    logits,
                    dim=-1
                )
                probabilities[:, PAD_TOKEN] = 0

                probabilities[:, START_TOKEN] = 0

                next_token = torch.multinomial(
                    probabilities,
                    num_samples=1
                ).item()

                if next_token == END_TOKEN:
                    break

                sequence.append(next_token)

            name = "".join(
                id_to_char[token]
                for token in sequence
                if token not in [
                    START_TOKEN,
                    END_TOKEN,
                    PAD_TOKEN
                ]
            )

            names.append(name)

    return names


