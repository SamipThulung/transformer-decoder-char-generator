from model.transformer_model import FinalModel
from utils.read_names import read_dino_csv
from utils.generate_names import predict_names
import string



block_size = 32 #total prediction
dropout=0.0
tokens = 20
embd = 4
vocab_size = 28
no_head = 2
head_dim = embd//no_head
total_labels = ['<start>']+[*string.ascii_lowercase] + ['<end>']

total_dataset = read_dino_csv("assets/dinosaurs.csv")

# Model initialize
model = FinalModel()


# Loss function
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

Epochs = 25

for ep in range(Epochs):
  random.shuffle(total_dataset)
  loss_val = 0
  length_total = len(total_dataset)
  for j, ent in enumerate(total_dataset):
    # print(ent)
    # print("loss: ", loss_val, end="\r")

    for ind in range(1, 20):
      if ent[ind] == 28:
        break
      entry = ent[:ind] + [28] * (20-len(ent[:ind]))
      out = ent[ind]
      # print(torch.unsqueeze(torch.Tensor(entry, dtype=torch.int),0))

      optimizer.zero_grad()

      # Make predictions for this batch
      outputs = model(torch.unsqueeze(torch.Tensor(entry),0).type(torch.int))

      # Calculate loss and back prop
      loss = loss_fn( outputs.view(1*1, 29), torch.Tensor([out]).type(torch.long))
      loss.backward()

      # Adjust learning weights
      optimizer.step()

      loss_val += loss.item()
      print(f"\rEpoch: {ep}/{Epochs}, loss: {loss.item()}, {j}/{length_total}", end="")

  print()
  print("Total epoch loss: ", loss_val)

print("Saving the model")
torch.save(model.state_dict(), "dino_name_predictor.pth")

names = predict_names(
    model,
    num_names=30,
    max_length=19,
    temperature=0.8
)

print("\nGenerated Names:\n")

for i, name in enumerate(names, 1):
    print(f"{i:02d}. {name}")



