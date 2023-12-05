# -*- coding: utf-8 -*-
import os
import torch
import random
import numpy as np
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt


def analysis(model, loader, args, epoch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    N    = [0 for _ in range(args.num_classes)]   # within class sample size
    mean = [0 for _ in range(args.num_classes)]
    Sw_cls   = [0 for _ in range(args.num_classes)]
    loss = 0
    n_correct = 0

    model.eval()
    criterion_summed = torch.nn.CrossEntropyLoss(reduction='sum')
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        with torch.no_grad():
            output, h = model(data, ret = 'of') #[B, C], [B, 512]
            loss += criterion_summed(output, target).item()

            net_pred = torch.argmax(output, dim=1)
            n_correct += torch.sum(net_pred == target).item()
            for c in range(args.num_classes):
                idxs = torch.where(target == c)[0]

                if len(idxs) > 0: #i.e. if there are samples of class c in this batch
                    h_c = h[idxs, :]
                    mean[c] += torch.sum(h_c, dim = 0) #CHW
                    N[c] += h_c.shape[0]


    M = torch.stack(mean).T       # [512, K]
    M = M / torch.tensor(N, device=M.device).unsqueeze(0) # [512, K]
    loss /= sum(N)
    acc = n_correct / sum(N)

    for batch_idx, (data, target) in enumerate(loader, start=1):
        data, target = data.to(device), target.to(device)
        with torch.no_grad():
            output, h = model(data, ret='of')  # [B, C], [B, 512]

        for c in range(args.num_classes):
            idxs = torch.where(target == c)[0]
            if len(idxs) > 0:  # If no class-c in this batch
                h_c = h[idxs, :]  # [B, 512]
                # update within-class cov
                z = h_c - mean[c].unsqueeze(0)  # [B, 512]
                cov = torch.matmul(z.unsqueeze(-1), z.unsqueeze(1))  # [B 512 1] [B 1 512] -> [B, 512, 512]
                Sw_cls[c] += torch.sum(cov, dim=0)  # [512, 512]

    # global mean
    muG = torch.mean(M, dim=1, keepdim=True)  # [512, C] -> [512, 1]

    M_ = M - muG  # [512, C]
    Sb = torch.matmul(M_, M_.T) / args.num_classes  # [512, C] [C, 512] -> [512, 512]

    # ============== NC1 ==============
    Sw_all = sum(Sw_cls) / sum(N)  # [512, 512]
    for c in range(args.num_classes):
        Sw_cls[c] = Sw_cls[c] / N[c]

    Sw = Sw_all.cpu().numpy()
    Sb = Sb.cpu().numpy()
    eigvec, eigval, _ = svds(Sb, k=args.num_classes - 1)
    inv_Sb = eigvec @ np.diag(1 / eigval) @ eigvec.T
    nc1 = np.trace(Sw @ Sb)
    nc1_cls = [np.trace(Sw_cls1.cpu().numpy() @ inv_Sb) for Sw_cls1 in Sw_cls]
    nc1_cls = np.array(nc1_cls)

    # ============== NC2: norm and cos ==============
    W = model.fc_cb.weight.detach().T       #  [512, C]
    M_norms = torch.norm(M_, dim=0)   #  [C]
    W_norms = torch.norm(W, dim=0)  #  [C]

    # angle between W
    W_nomarlized = W / W_norms  # [512, C]
    cos = (W_nomarlized.T @ W_nomarlized).cpu().numpy()  # [C, D] [D, C] -> [C, C]
    cos_avg = (cos.sum(1) - np.diag(cos)) / (cos.shape[1] - 1)


     # angle between H
    M_normalized = M_ / M_norms  # [512, C]
    h_cos = (M_normalized.T @ M_normalized).cpu().numpy()
    h_cos_avg = (h_cos.sum(1) - np.diag(h_cos)) / (h_cos.shape[1] - 1)


    # angle between W and H	    # angle between W and H
    wh_cos = torch.sum(W_nomarlized*M_normalized, dim=0).cpu().numpy()  # [C]




    ''' Original draw plot methods without using wandb are as follows >>>>
    #==========================draw W==========================================
    plot_dir = "/scratch/hy2611/GLMC/VS_plot"
    os.makedirs(plot_dir, exist_ok=True)
    
        
    np.fill_diagonal(cos, 0)  

    plt.figure(figsize=(10, 8))
    cax = plt.imshow(cos, cmap='coolwarm', interpolation='nearest')
    plt.colorbar(cax)
    plt.title("Cosine Similarity between Class Weights")
    plt.xlabel("Class Index")
    plt.ylabel("Class Index")

    
    for i in range(cos.shape[0]):
        for j in range(cos.shape[1]):
            plt.text(j, i, f'{cos[i, j]:.2f}', ha='center', va='center', color='black')

    plot_path = os.path.join(plot_dir, f"cos_VS_5000_50_epoch_{epoch}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved cosine similarity heatmap to: {plot_path}")
    
    #==================draw relationship between W_i and h_i ==================
    
    # Normalize the means for each class
    H_normalized = [mean[c] / N[c] for c in range(args.num_classes)]
    
    cosine_similarities = [torch.nn.functional.cosine_similarity(W[i].unsqueeze(0), H_normalized[i].unsqueeze(0)).item()
                           for i in range(args.num_classes)]
    
    plt.figure(figsize=(10, 5))
    plt.bar(range(args.num_classes), cosine_similarities)
    plt.xlabel('Class Index')
    plt.ylabel('Cosine Similarity')
    plt.title('Cosine Similarity between Class Weights and Mean Features')
    relationship_plot_path = os.path.join(plot_dir, f"Wi_Hi_cos_VS_5000_50_epoch_{epoch}.png")
    plt.savefig(relationship_plot_path)
    plt.close()
    
    print(f"Saved Wi and Hi cosine similarity plot to: {relationship_plot_path}")
    #==========================================================================
    '''






    return {
        "loss": loss,
        "acc": acc,
        "nc1": nc1,
        'nc1_cls': nc1_cls,
        "w_norm": W_norms.cpu().numpy(),
        "h_norm": M_norms.cpu().numpy(),
        "w_cos": cos,
        "w_cos_avg": cos_avg,
        "h_cos":h_cos,
        "h_cos_avg": h_cos_avg,
        "wh_cos": wh_cos
    }

