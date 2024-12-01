import graph

# Usage
dataset = [
    {'input': [1, 2, 3], 'output': [2, 3, 4]},
    {'input': [2, 3, 4], 'output': [3, 4, 5]}
]

# Initialize model
gpt = graph(
    vocab_size=12,
    embed_size=16,
    num_heads=2,
    num_layers=2,
    context_length=3
)

# Train model
gpt.train(dataset, batch_size=2, epochs=50)

# Generate predictions
input_seq = [4, 5, 6]
predicted_seq = gpt.predict(input_seq, max_length=4)
print(f"Input: {input_seq}")
print(f"Predicted: {predicted_seq}")

# Save model
# gpt.save("gpt_model.pt")
gpt.export('text.json')
