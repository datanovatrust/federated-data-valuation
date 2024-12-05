# src/utils/peft_utils.py

def freeze_base_model(model):
    for param in model.parameters():
        param.requires_grad = False

def unfreeze_lora_parameters(model):
    for name, param in model.named_parameters():
        if 'lora' in name:
            param.requires_grad = True
