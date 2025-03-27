"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
from torch.utils.data._utils.collate import default_collate


def lengths_to_mask(lengths, max_len):
    mask = torch.arange(max_len, device=lengths.device).expand(
        len(lengths), max_len
    ) < lengths.unsqueeze(1)
    return mask


def collate_tensors(batch):
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


## social collate
def collate_v2(batch):
    notnone_batches = [b for b in batch if b is not None]
    databatch = [b["inp"] for b in notnone_batches]
    missingbatch = [b["missing"] for b in notnone_batches]
    audiobatch = [b["audio"] for b in notnone_batches]
    lenbatch = [b["lengths"] for b in notnone_batches]
    alenbatch = [b["audio_lengths"] for b in notnone_batches]
    keyframebatch = [b["keyframes"] for b in notnone_batches]
    klenbatch = [b["key_lengths"] for b in notnone_batches]
    
    # Handle text embeddings if available
    if "text_embedding" in notnone_batches[0]:
        text_embedding_batch = [b["text_embedding"] for b in notnone_batches]
        text_embedding_len_batch = [b["text_embedding_lengths"] for b in notnone_batches]
        has_text_embeddings = True
    else:
        has_text_embeddings = False

    databatchTensor = collate_tensors(databatch)
    missingbatchTensor = collate_tensors(missingbatch)
    audiobatchTensor = collate_tensors(audiobatch)
    lenbatchTensor = torch.as_tensor(lenbatch)
    alenbatchTensor = torch.as_tensor(alenbatch)
    keyframeTensor = collate_tensors(keyframebatch)
    klenbatchTensor = torch.as_tensor(klenbatch)
    
    # Collate text embeddings if available
    if has_text_embeddings:
        text_embedding_tensor = collate_tensors(text_embedding_batch)
        text_embedding_len_tensor = torch.as_tensor(text_embedding_len_batch)

    maskbatchTensor = (
        lengths_to_mask(lenbatchTensor, databatchTensor.shape[-1])
        .unsqueeze(1)
        .unsqueeze(1)
    )  # unqueeze for broadcasting
    motion = databatchTensor
    cond = {
        "y": {
            "missing": missingbatchTensor,
            "mask": maskbatchTensor,
            "lengths": lenbatchTensor,
            "audio": audiobatchTensor,
            "alengths": alenbatchTensor,
            "keyframes": keyframeTensor,
            "klengths": klenbatchTensor,
        }
    }
    
    # Add text embeddings to condition dict if available
    if has_text_embeddings:
        cond["y"]["text_embedding"] = text_embedding_tensor
        cond["y"]["text_embedding_lengths"] = text_embedding_len_tensor
        
    return motion, cond


def social_collate(batch):
    adapted_batch = []
    for b in batch:
        item = {
            "inp": torch.tensor(b["motion"].T).to(torch.float32).unsqueeze(1),
            "lengths": b["m_length"],
            "audio": b["audio"]
            if torch.is_tensor(b["audio"])
            else torch.tensor(b["audio"]).to(torch.float32),
            "keyframes": torch.tensor(b["keyframes"]).to(torch.float32),
            "key_lengths": b["k_length"],
            "audio_lengths": b["a_length"],
            "missing": torch.tensor(b["missing"]).to(torch.float32),
        }
        
        # Add text embedding if available
        if "text_embedding" in b:
            item["text_embedding"] = torch.tensor(b["text_embedding"]).to(torch.float32)
            item["text_embedding_lengths"] = b["t_length"]
            
        adapted_batch.append(item)
        
    return collate_v2(adapted_batch)
