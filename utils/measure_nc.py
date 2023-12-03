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
    Sw   = [0 for _ in range(args.num_classes)]
    loss = 0
    n_correct = 0

    model.eval()
    criterion_summed = torch.nn.CrossEntropyLoss(reduction='sum')
    for computation in ['Mean', 'Cov']:
        for batch_idx, (data, target) in enumerate(loader, start=1):
            if isinstance(data, list): 
                data = data[0]
            data, target = data.to(device), target.to(device)

            with torch.no_grad():
                output, h = model(data, ret='of')  # [B, C], [B, 512]

            for c in range(args.num_classes):
                idxs = (target == c).nonzero(as_tuple=True)[0]
                if len(idxs) == 0:  # If no class-c in this batch
                    continue
                h_c = h[idxs, :]    # [B, 512]

                # update class means
                if computation == 'Mean':
                    N[c] += h_c.shape[0]
                    mean[c] += torch.sum(h_c, dim=0)
                    loss += criterion_summed(output, target).item()  # during calculation of class means, calculate loss

                # update within-class cov
                elif computation == 'Cov':
                    z = h_c - mean[c].unsqueeze(0)  # [B, 512]
                    cov = torch.matmul(z.unsqueeze(-1), z.unsqueeze(1))   # [B 512 1] [B 1 512] -> [B, 512, 512]
                    Sw += torch.sum(cov, dim=0)     # [512, 512]

                    # during calculation of within-class covariance, calculate network's accuracy
                    net_pred = torch.argmax(output[idxs, :], dim=1)
                    n_correct += sum(net_pred == target[idxs]).item()

        if computation == 'Mean':
            for c in range(args.num_classes):
                mean[c] /= N[c]
                M = torch.stack(mean).T
        elif computation == 'Cov':
            Sw /= sum(N)

    loss /= sum(N)
    acc = n_correct / sum(N)

    # between-class covariance
    muG = torch.mean(M, dim=1, keepdim=True)  # global mean: [512, C]
    M_ = M - muG  # [512, C]
    Sb = torch.matmul(M_, M_.T) / args.num_classes

    # avg norm
    W = model.fc_cb.weight.detach()       # [C, 512]
    M_norms = torch.norm(M_, dim=0)   # [C]
    W_norms = torch.norm(W.T, dim=0)  # [C]
    #h_norm_cov = (torch.std(M_norms) / torch.mean(M_norms)).item()
    #w_norm_cov = (torch.std(W_norms) / torch.mean(W_norms)).item()

    # nc1 = tr{Sw Sb^-1}
    Sw = Sw.cpu().numpy()
    Sb = Sb.cpu().numpy()
    eigvec, eigval, _ = svds(Sb, k=args.num_classes - 1)
    inv_Sb = eigvec @ np.diag(eigval ** (-1)) @ eigvec.T
    nc1 = np.trace(Sw @ inv_Sb)

    # mutual coherence
    W_nomarlized = W.T / W_norms # [512, C]
    cos = ( W_nomarlized.T @ W_nomarlized ).cpu().numpy()  # [C, D] [D, C] -> [C, C]
    cos_avg = (cos.sum(1) - np.diag(cos)) / (cos.shape[1] - 1)

    M_normalized = M_ / M_norms  # [512, C]
    cos_wh = torch.sum(W_nomarlized*M_normalized, dim=0).cpu().numpy()  # [C]
    
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
    






    return {
        "loss": loss,
        "acc": acc,
        "nc1": nc1,
        "w_norm": W_norms.cpu().numpy(),
        "h_norm": M_norms.cpu().numpy(),
        "w_cos": cos,
        "w_cos_avg": cos_avg,
        "wh_cos": cos_wh
    }

