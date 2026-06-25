#  Strings AI Core Engine
#  Copyright (C) 2026  Avaneesh Shahi
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
import torch
import torch.nn as nn
import torch.optim as optim

# =====================================================================
#                      PROJECT: STRINGS (v0.0.1)
#           Baseline Multi-Sequence Predictive Neural Network
# =====================================================================

# 1. CORE DATASET SEEDING
# Strings is initialized on two distinct conversational baseline sequences
dataset = [
    "the color of sky is blue",
    "the color of banana is yellow"
]

# Build a unique, zero-indexed vocabulary from all provided data strings
vocab = set()
for sentence in dataset:
    vocab.update(sentence.lower().split())
vocab = list(sorted(vocab))

word_to_idx = {word: i for i, word in enumerate(vocab)}
idx_to_word = {i: word for i, word in enumerate(vocab)}
vocab_size = len(vocab)

# Parse inputs into historical context chains and matching target tokens
inputs = []
targets = []

for sentence in dataset:
    words = sentence.lower().split()
    for i in range(1, len(words)):
        context = [word_to_idx[w] for w in words[:i]]
        target = word_to_idx[words[i]]
        inputs.append(context)
        targets.append(target)


# =====================================================================
# 2. NEURAL NETWORK ARCHITECTURE DEFINITION
# =====================================================================
class StringsCoreModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # Continuous vector space embedding layer for character sequences
        self.embedding = nn.Embedding(vocab_size, 16)
        # Recurrent layer serving as structural context sequence memory
        self.rnn = nn.RNN(input_size=16, hidden_size=16, batch_first=True)
        # Final fully-connected projection mapping back to vocabulary dimensions
        self.output = nn.Linear(16, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        out, hidden = self.rnn(embedded)
        # Extract the final temporal state node from the evaluation sequence
        last_word_output = out[:, -1, :]
        return self.output(last_word_output)


# Initialize the primary network weights and optimization components
Strings = StringsCoreModel(vocab_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(Strings.parameters(), lr=0.01)


# =====================================================================
# 3. WEIGHT OPTIMIZATION AND TRAINING PHASE
# =====================================================================
print("==================================================")
print("  Initializing Training Loop for: STRINGS v0.0.1   ")
print("==================================================")

# Execute backpropagation epochs to tune internal connection matrices
for epoch in range(401):
    total_loss = 0
    for context, target in zip(inputs, targets):
        input_tensor = torch.tensor([context], dtype=torch.long)
        target_tensor = torch.tensor([target], dtype=torch.long)

        output = Strings(input_tensor)
        loss = criterion(output, target_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if epoch % 100 == 0:
        print(f"Epoch {epoch:3d} | Vector Matrix Error (Loss): {total_loss / len(inputs):.4f}")

print("\n[Optimization Complete] Strings internal weights locked and configured.")
print("==================================================\n")


# =====================================================================
# 4. INTERACTIVE USER TERMINAL ENVIRONMENT (INFERENCE)
# =====================================================================
print("--- STRINGS v0.0.1 LIVE CHAT INTERFACE ---")
print("Provide a prompt phrase (e.g., 'The color of sky is')")
print("Type 'exit' to terminate the process.\n")

while True:
    user_prompt = input("You: ")
    if user_prompt.lower().strip() == 'exit':
        print("Shutting down Strings v0.0.1 instance...")
        break

    prompt_words = user_prompt.lower().strip().split()

    # Pass elements through the structural validation checker
    context_indices = []
    unknown_word_found = False

    for w in prompt_words:
        if w in word_to_idx:
            context_indices.append(word_to_idx[w])
        else:
            print(f"Strings: Token '{w}' falls outside the active training parameters.")
            unknown_word_found = True
            break

    if unknown_word_found or len(context_indices) == 0:
        continue

    # Feed validated text sequence straight into the network layers
    input_tensor = torch.tensor([context_indices], dtype=torch.long)

    with torch.no_grad():
        prediction = Strings(input_tensor)
        predicted_idx = torch.argmax(prediction, dim=1).item()
        predicted_word = idx_to_word[predicted_idx]

    print(f"Strings: {user_prompt.strip()} {predicted_word}\n")