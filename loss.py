import torch
import torch.nn.functional as func
from torch import nn


def balanced_entropy(preds, targets):
    eps = 1e-6
    soft = nn.Softmax(dim=1)
    z_preds = soft(preds)
    cliped_z = torch.clamp(z_preds, eps, 1 - eps)
    log_z = torch.log(cliped_z)
    num_classes = targets.size(1)
    ind = torch.argmax(targets, 1).type(torch.int)

    total = torch.sum(targets)

    m_c, n_c = [], []
    for class_num in range(num_classes):
        m_c.append((ind == class_num).type(torch.int))
        n_c.append(torch.sum(m_c[-1]).type(torch.float))

    c_list = []
    for i in range(num_classes):
        c_list.append(total - n_c[i])
    tc = sum(c_list)

    loss = 0
    for i in range(num_classes):
        weight = c_list[i] / tc
        m_c_one_hot = func.one_hot(
            (i * m_c[i]).permute(1, 2, 0).type(torch.long), num_classes
        )
        m_c_one_hot = m_c_one_hot.permute(2, 3, 0, 1)
        y_c = m_c_one_hot * targets
        loss += weight * torch.sum(-torch.sum(y_c * log_z, axis=2))
    return loss / num_classes


def cross_two_tasks_weight(rooms, boundaries):
    p1 = torch.sum(rooms).type(torch.float)
    p2 = torch.sum(boundaries).type(torch.float)
    w1 = torch.div(p2, p1 + p2)
    w2 = torch.div(p1, p1 + p2)
    return w1, w2
