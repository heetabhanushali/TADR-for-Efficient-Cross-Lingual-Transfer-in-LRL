# After creating the model, test it:
print("\n🧪 Quick sanity check...")

# Get a batch
for batch in train_loader:
    input_ids = batch['input_ids'][:4].to(DEVICE)
    labels = batch['labels'][:4].to(DEVICE)
    lang_ids = batch['language_ids'][:4]
    attention_mask = batch['attention_mask'][:4].to(DEVICE)
    
    # Forward pass
    outputs = model(lang_ids=lang_ids, input_ids=input_ids, attention_mask=attention_mask)
    loss = torch.nn.functional.cross_entropy(outputs['logits'], labels)
    
    print(f"Initial loss: {loss.item():.4f}")
    print(f"Logits range: {outputs['logits'].min().item():.3f} to {outputs['logits'].max().item():.3f}")
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    print(f"Gradient norm: {grad_norm:.4f}")
    
    if grad_norm < 0.001:
        print("❌ WARNING: Gradients too small!")
    elif grad_norm > 100:
        print("❌ WARNING: Gradients exploding!")
    else:
        print("✅ Gradients look OK")
    
    break

print("\n✅ Sanity check complete\n")