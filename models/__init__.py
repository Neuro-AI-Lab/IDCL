from .model import Transformer_Based_Model

def get_model(model_name, args, D_text, D_audio, n_classes, device):
    """
    Factory function to initialize and return the requested model.
    """
    model_name = model_name.lower()
    
    if model_name == 'transformer':
        model = Transformer_Based_Model(
            D_text=D_text, D_audio=D_audio,
            n_classes=n_classes, hidden_dim=args.hidden_dim, dropout=args.dropout
        )

    # add yout own model here
    # elif model_name == 'your_model':
    #     model = YourModel(
    #         D_text=D_text, D_audio=D_audio,
    #         n_classes=n_classes, hidden_dim=args.hidden_dim, dropout=args.dropout
    #     )

    else:
        raise ValueError(f"Unknown model name: {model_name}. Choose from 'transformer'")
        
    return model.to(device)
