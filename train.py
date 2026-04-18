import os
import argparse
import time
import random
import json
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score, classification_report
from datetime import datetime

from dataloader.dataloader import IEMOCAPDataset, MELDDataset
from models import get_model
from losses.IDCL import IDCL
from losses.CE import MaskedNLLLoss

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_dataloaders(dataset_name, batch_size):
    if dataset_name.upper() == 'IEMOCAP':
        train_dataset = IEMOCAPDataset(split='train')
        valid_dataset = IEMOCAPDataset(split='dev')
        test_dataset = IEMOCAPDataset(split='test')

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=valid_dataset.collate_fn)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=test_dataset.collate_fn)

        D_audio, D_visual, D_text = 100, 512, 768
        n_classes = 6
        n_speakers = 2
        return train_loader, valid_loader, test_loader, D_audio, D_visual, D_text, n_classes, n_speakers

    elif dataset_name.upper() == 'MELD':
        all_dataset = MELDDataset('./data/MELD/MELD_features_raw1.pkl', all=True)
        
        train_loader = torch.utils.data.DataLoader(all_dataset, batch_size=len(all_dataset), collate_fn=all_dataset.collate_fn, shuffle=True)
        valid_loader = train_loader  # pretrain only
        test_loader = train_loader

        D_audio, D_visual, D_text = 300, 342, 600
        n_classes = 7
        n_speakers = 9
        return train_loader, valid_loader, test_loader, D_audio, D_visual, D_text, n_classes, n_speakers
    else:
        raise ValueError("Unsupported dataset")


def train_or_eval(model, ce_loss_fn, idcl_loss_fn, dataloader, optimizer=None, args=None, is_train=False, n_classes=4, apply_idcl=False, apply_ce=True):
    losses, preds, labels, masks = [], [], [], []
    model.train() if is_train else model.eval()

    device = next(model.parameters()).device

    for data in dataloader:
        if is_train:
            optimizer.zero_grad()

        textf, visuf, acouf, qmask, umask, label = data[:-1]
        textf = torch.nan_to_num(textf, nan=0.0).to(device)
        acouf = torch.nan_to_num(acouf, nan=0.0).to(device)
        qmask = qmask.permute(1, 0, 2).to(device)
        umask = umask.to(device)
        label = label.to(device)

        all_log_prob, t_feat, a_feat, _ = model(textf, acouf, umask, qmask)

        if apply_ce:
            loss = ce_loss_fn(all_log_prob.view(-1, n_classes), label.view(-1), umask)
        else:
            loss = torch.tensor(0.0, device=device)

        if apply_idcl:
            loss_ta = idcl_loss_fn(a_feat, t_feat)
            loss_at = idcl_loss_fn(t_feat, a_feat)
            loss_D = (loss_ta + loss_at) / 2.0
            loss = loss + args.idcl_weight * loss_D

        preds.append(torch.argmax(all_log_prob.view(-1, n_classes), 1).cpu().numpy())
        labels.append(label.view(-1).cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())
        losses.append(loss.item() * masks[-1].sum())

        if is_train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

    preds, labels, masks = map(np.concatenate, [preds, labels, masks])
    avg_loss = round(np.sum(losses) / np.sum(masks), 4)
    avg_accuracy = round(accuracy_score(labels, preds, sample_weight=masks) * 100, 2)
    avg_fscore = round(f1_score(labels, preds, sample_weight=masks, average='weighted') * 100, 2)

    return avg_loss, avg_accuracy, avg_fscore, labels, preds, masks


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='transformer', choices=['transformer'])
    parser.add_argument('--loss_type', type=str, default='ce+idcl', choices=['ce+idcl'])
    parser.add_argument('--idcl_weight', type=float, default=0.05)
    parser.add_argument('--pretrain_K', type=int, default=15)
    parser.add_argument('--finetune_K', type=int, default=15)
    
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--pretrain_epochs', type=int, default=1000) # MELD pretraining
    parser.add_argument('--finetune_epochs', type=int, default=200) # IEMOCAP finetuning
    parser.add_argument('--seed', type=int, default=20260505)

    args = parser.parse_args()
    set_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{datetime.now()}] Setup: Model={args.model}, Loss={args.loss_type}")

    ce_loss_fn = MaskedNLLLoss()
    pretrain_idcl_fn = IDCL(K=args.pretrain_K, temperature=1.0)
    finetune_idcl_fn = IDCL(K=args.finetune_K, temperature=0.10)

    # ----------------------------------------------------
    # STAGE 1: PRETRAINING (MELD)
    # ----------------------------------------------------
    if args.pretrain_epochs > 0:
        if os.path.exists(f"pretrained_{args.model}_idcl.pth"):
            print(f"[*] Found existing pretrain weights for '{args.model}'. Skipping STAGE 1.")
        else:
            print("\n" + "="*50)
            print("STAGE 1: PRETRAINING (MELD)")
            print("="*50)
            p_train_loader, p_valid_loader, p_test_loader, p_D_audio, p_D_visual, p_D_text, p_classes, p_spk = get_dataloaders('MELD', args.batchsize)
            
            pretrain_model = get_model(args.model, args, p_D_text, p_D_audio, p_classes, device)
            p_optimizer = optim.Adam(pretrain_model.parameters(), lr=args.lr, weight_decay=1e-4)
    
            for epoch in range(args.pretrain_epochs):

                tr_loss, _, _, _, _, _ = train_or_eval(pretrain_model, ce_loss_fn, pretrain_idcl_fn, p_train_loader, p_optimizer, args, is_train=True, n_classes=p_classes, apply_idcl=True, apply_ce=False)
                
                print(f"Pretrain Epoch {epoch+1:03d}/{args.pretrain_epochs} | Train Loss: {tr_loss:.4f}")

            torch.save(pretrain_model.state_dict(), f"pretrained_{args.model}_idcl.pth")
            print(f"Saved pretrained weights: pretrained_{args.model}_idcl.pth")

    # ----------------------------------------------------
    # STAGE 2: FINETUNING (IEMOCAP)
    # ----------------------------------------------------
    print("\n" + "="*50)
    print("STAGE 2: FINETUNING (IEMOCAP)")
    print("="*50)
    f_train_loader, f_valid_loader, f_test_loader, f_D_audio, f_D_visual, f_D_text, f_classes, f_spk = get_dataloaders('IEMOCAP', args.batchsize)
    
    finetune_model = get_model(args.model, args, f_D_text, f_D_audio, f_classes, device)

    # Load pretrained weights
    if args.pretrain_epochs > 0 and os.path.exists(f"pretrained_{args.model}_idcl.pth"):
        pretrained_dict = torch.load(f"pretrained_{args.model}_idcl.pth", map_location=device)
        model_dict = finetune_model.state_dict()
        # Filter out projection layers and classifier which have different shapes
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        finetune_model.load_state_dict(model_dict, strict=False)
        print(f"Loaded {len(pretrained_dict)} layers from pretrained model.")

    f_optimizer = optim.Adam(finetune_model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_valid_fscore = -1.0
    best_test_fscore = 0.0
    best_test_acc = 0.0
    best_report = {}

    use_idcl_finetune = (args.loss_type == 'ce+idcl')

    for epoch in range(args.finetune_epochs):
        tr_loss, tr_acc, tr_f1, _, _, _ = train_or_eval(finetune_model, ce_loss_fn, finetune_idcl_fn, f_train_loader, f_optimizer, args, is_train=True, n_classes=f_classes, apply_idcl=use_idcl_finetune)
        val_loss, val_acc, val_f1, _, _, _ = train_or_eval(finetune_model, ce_loss_fn, finetune_idcl_fn, f_valid_loader, args=args, is_train=False, n_classes=f_classes, apply_idcl=use_idcl_finetune)
        t_loss, t_acc, t_f1, t_labels, t_preds, t_masks = train_or_eval(finetune_model, ce_loss_fn, finetune_idcl_fn, f_test_loader, args=args, is_train=False, n_classes=f_classes, apply_idcl=use_idcl_finetune)

        print(f"Finetune Epoch {epoch+1:03d}/{args.finetune_epochs} | Train Loss: {tr_loss:.4f} Acc: {tr_acc:.2f} | Valid Acc: {val_acc:.2f} | Test Acc: {t_acc:.2f} F1: {t_f1:.2f}")

        if val_f1 > best_valid_fscore:
            best_valid_fscore = val_f1
            best_test_fscore = t_f1
            best_test_acc = t_acc
            best_report = classification_report(t_labels, t_preds, sample_weight=t_masks, digits=4, output_dict=True)

    print("-" * 50)
    print(f"[{args.model} | {args.loss_type}] Best Test F-score: {best_test_fscore:.2f} | Best Test Acc: {best_test_acc:.2f}")

    os.makedirs("results", exist_ok=True)
    res_path = f"results/IEMOCAP_{args.model}_{args.loss_type}_twostage.json"
    with open(res_path, 'w') as f:
        json.dump({
            "model": args.model,
            "loss_type": args.loss_type,
            "best_test_fscore": best_test_fscore,
            "best_test_acc": best_test_acc,
            "classification_report": best_report
        }, f, indent=4)
        
    print(f"Results saved to {res_path}")

