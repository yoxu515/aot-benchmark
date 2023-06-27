import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def foreground2background(dis, obj_num):
    if obj_num == 1:
        return dis
    bg_dis = []
    for i in range(obj_num):
        obj_back = []
        for j in range(obj_num):
            if i == j:
                continue
            obj_back.append(dis[j].unsqueeze(0))
        obj_back = torch.cat(obj_back, dim=1)
        obj_back, _ = torch.min(obj_back, dim=1, keepdim=True)
        bg_dis.append(obj_back)
    bg_dis = torch.cat(bg_dis, dim=0)
    return bg_dis

WRONG_LABEL_PADDING_DISTANCE = 5e4
#############################################################GLOBAL_DIST_MAP
def _pairwise_distances(x, x2, y, y2):
    """
    Computes pairwise squared l2 distances between tensors x and y.
    Args:
    x: [n, feature_dim].
    y: [m, feature_dim].
    Returns:
    d: [n, m].
    """
    xs = x2
    ys = y2

    xs = xs.unsqueeze(1)
    ys = ys.unsqueeze(0)
    d = xs + ys - 2. * torch.matmul(x, torch.t(y))
    return d

##################
def _flattened_pairwise_distances(reference_embeddings, ref_square, query_embeddings, query_square):
    """
    Calculates flattened tensor of pairwise distances between ref and query.
    Args:
        reference_embeddings: [..., embedding_dim],
          the embedding vectors for the reference frame
        query_embeddings: [..., embedding_dim], 
          the embedding vectors for the query frames.
    Returns:
        dists: [reference_embeddings.size / embedding_dim, query_embeddings.size / embedding_dim]
    """
    dists = _pairwise_distances(query_embeddings, query_square, reference_embeddings, ref_square)
    return dists

def _nn_features_per_object_for_chunk(
    reference_embeddings, ref_square, query_embeddings, query_square, wrong_label_mask):
    """Extracts features for each object using nearest neighbor attention.
    Args:
        reference_embeddings: [n_chunk, embedding_dim],
          the embedding vectors for the reference frame.
        query_embeddings: [m_chunk, embedding_dim],
          the embedding vectors for the query frames.
        wrong_label_mask: [n_objects, n_chunk],
          the mask for pixels not used for matching.
    Returns:
        nn_features: A float32 tensor of nearest neighbor features of shape
          [m_chunk, n_objects, n_chunk].
    """
    if reference_embeddings.dtype == torch.float16:
        wrong_label_mask = wrong_label_mask.half()
    else:
        wrong_label_mask = wrong_label_mask.float()

    reference_embeddings_key = reference_embeddings
    query_embeddings_key = query_embeddings
    dists = _flattened_pairwise_distances(reference_embeddings_key, ref_square, query_embeddings_key, query_square)
    
    dists = (torch.unsqueeze(dists, 1) +
            torch.unsqueeze(wrong_label_mask, 0) *
           WRONG_LABEL_PADDING_DISTANCE)
    
    features, _ = torch.min(dists, 2, keepdim=True)
    return features

def _nearest_neighbor_features_per_object_in_chunks(
    reference_embeddings_flat, query_embeddings_flat, reference_labels_flat, n_chunks):
    """Calculates the nearest neighbor features per object in chunks to save mem.
    Uses chunking to bound the memory use.
    Args:
        reference_embeddings_flat: [n, embedding_dim],
          the embedding vectors for the reference frame.
        query_embeddings_flat: [m, embedding_dim], 
          the embedding vectors for the query frames.
        reference_labels_flat: [n, n_objects], 
          the class labels of the reference frame.
        n_chunks: Integer, the number of chunks to use to save memory
          (set to 1 for no chunking).
    Returns:
        nn_features: [m, n_objects, n].
    """

    feature_dim, embedding_dim = query_embeddings_flat.size()
    chunk_size = int(np.ceil(float(feature_dim) / n_chunks))
    wrong_label_mask = reference_labels_flat < 0.1
    wrong_label_mask = wrong_label_mask.permute(1, 0)
    ref_square = reference_embeddings_flat.pow(2).sum(1)
    query_square = query_embeddings_flat.pow(2).sum(1)

    all_features = []
    for n in range(n_chunks):
        if n_chunks == 1:
            query_embeddings_flat_chunk = query_embeddings_flat
            query_square_chunk = query_square
            chunk_start = 0
        else:
            chunk_start = n * chunk_size
            chunk_end = (n + 1) * chunk_size
            query_square_chunk = query_square[chunk_start:chunk_end]
            if query_square_chunk.size(0) == 0:
                continue
            query_embeddings_flat_chunk = query_embeddings_flat[chunk_start:chunk_end]
        features = _nn_features_per_object_for_chunk(
            reference_embeddings_flat, ref_square, query_embeddings_flat_chunk, query_square_chunk,
            wrong_label_mask)
        all_features.append(features)
    if n_chunks == 1:
        nn_features = all_features[0]
    else:
        nn_features = torch.cat(all_features, dim=0)
    

    return nn_features


def global_matching(
    reference_embeddings, query_embeddings, reference_labels,
    n_chunks=100, dis_bias=0., ori_size=None, atrous_rate=1, use_float16=True):
    """
    Calculates the distance to the nearest neighbor per object.
    For every pixel of query_embeddings calculate the distance to the
    nearest neighbor in the (possibly subsampled) reference_embeddings per object.
    Args:
        reference_embeddings: [height, width, embedding_dim],
          the embedding vectors for the reference frame.
        query_embeddings: [height, width,
          embedding_dim], the embedding vectors for the query frames.
        reference_labels: [height, width, obj_nums], 
          the class labels of the reference frame.
        n_chunks: Integer, the number of chunks to use to save memory
          (set to 1 for no chunking).
        dis_bias: [n_objects], foreground and background bias
        ori_size: (ori_height, ori_width),
          the original spatial size. If "None", (ori_height, ori_width) = (height, width).
        atrous_rate: Integer, the atrous rate of reference_embeddings.
        use_float16: Bool, if "True", use float16 type for matching.
    Returns:
        nn_features: [n_query_images, ori_height, ori_width, n_objects, feature_dim].
    """
    
    assert (reference_embeddings.size()[:2] == reference_labels.size()[:2])
    if use_float16:
        query_embeddings = query_embeddings.half()
        reference_embeddings = reference_embeddings.half()
    h, w, _ = query_embeddings.size()
    obj_nums = reference_labels.size(2)
    if atrous_rate > 1:
        h_pad = (atrous_rate - h % atrous_rate) % atrous_rate
        w_pad = (atrous_rate - w % atrous_rate) % atrous_rate
        if h_pad > 0  or w_pad > 0:
            reference_embeddings = F.pad(reference_embeddings, (0, 0, 0, w_pad, 0, h_pad))
            reference_labels = F.pad(reference_labels, (0, 0, 0, w_pad, 0, h_pad))

        reference_embeddings = reference_embeddings.view((h + h_pad) // atrous_rate, atrous_rate, 
                                                         (w + w_pad) // atrous_rate, atrous_rate, -1)
        reference_labels = reference_labels.view((h + h_pad) // atrous_rate, atrous_rate, 
                                                         (w + w_pad) // atrous_rate, atrous_rate, -1)
        reference_embeddings = reference_embeddings[:, 0, :, 0, :].contiguous()
        reference_labels = reference_labels[:, 0, :, 0, :].contiguous()
    
    embedding_dim = query_embeddings.size()[-1]
    query_embeddings_flat = query_embeddings.view(-1, embedding_dim)
    reference_embeddings_flat = reference_embeddings.view(-1, embedding_dim)
    reference_labels_flat = reference_labels.view(-1, obj_nums)
    

    all_ref_fg = torch.sum(reference_labels_flat, dim=1, keepdim=True) > 0.9
    reference_labels_flat = torch.masked_select(reference_labels_flat, 
        all_ref_fg.expand(-1, obj_nums)).view(-1, obj_nums)
    if reference_labels_flat.size(0) == 0:
        return torch.ones(1, h, w, obj_nums, 1, device=all_ref_fg.device)
    reference_embeddings_flat = torch.masked_select(reference_embeddings_flat, 
        all_ref_fg.expand(-1, embedding_dim)).view(-1, embedding_dim)


    nn_features = _nearest_neighbor_features_per_object_in_chunks(
        reference_embeddings_flat, query_embeddings_flat, reference_labels_flat,
        n_chunks)

    nn_features_reshape = nn_features.view(1, h, w, obj_nums, 1)
    nn_features_reshape = (torch.sigmoid(nn_features_reshape + dis_bias.view(1, 1, 1, -1, 1)) - 0.5) * 2

    if ori_size is not None:
        nn_features_reshape = nn_features_reshape.view(h, w, obj_nums, 1).permute(2, 3, 0, 1)
        nn_features_reshape = F.interpolate(nn_features_reshape, size=ori_size, 
            mode='bilinear', align_corners=True).permute(2, 3, 0, 1).view(1, ori_size[0], ori_size[1], obj_nums, 1)

    if use_float16:
        nn_features_reshape = nn_features_reshape.float()
    return nn_features_reshape

def global_matching_for_eval(
    all_reference_embeddings, query_embeddings, all_reference_labels,
    n_chunks=20, dis_bias=0., ori_size=None, atrous_rate=1, use_float16=True):
    """
    Calculates the distance to the nearest neighbor per object.
    For every pixel of query_embeddings calculate the distance to the
    nearest neighbor in the (possibly subsampled) reference_embeddings per object.
    Args:
        all_reference_embeddings: A list of reference_embeddings,
          each with size [height, width, embedding_dim],
          the embedding vectors for the reference frame.
        query_embeddings: [n_query_images, height, width,
          embedding_dim], the embedding vectors for the query frames.
        all_reference_labels: A list of reference_labels,
          each with size [height, width, obj_nums], 
          the class labels of the reference frame.
        n_chunks: Integer, the number of chunks to use to save memory
          (set to 1 for no chunking).
        dis_bias: [n_objects], foreground and background bias
        ori_size: (ori_height, ori_width),
          the original spatial size. If "None", (ori_height, ori_width) = (height, width).
        atrous_rate: Integer, the atrous rate of reference_embeddings.
        use_float16: Bool, if "True", use float16 type for matching.
    Returns:
        nn_features: [n_query_images, ori_height, ori_width, n_objects, feature_dim].
    """
    
    h, w, _ = query_embeddings.size()
    obj_nums = all_reference_labels[0].size(2)
    all_reference_embeddings_flat = []
    all_reference_labels_flat = []
    embedding_dim = query_embeddings.size()[-1]
    ref_num = len(all_reference_labels)
    n_chunks *= ref_num
    if ref_num == 1:
        reference_embeddings, reference_labels = all_reference_embeddings[0], all_reference_labels[0]
        if atrous_rate > 1:
            h_pad = (atrous_rate - h % atrous_rate) % atrous_rate
            w_pad = (atrous_rate - w % atrous_rate) % atrous_rate
            if h_pad > 0  or w_pad > 0:
                reference_embeddings = F.pad(reference_embeddings, (0, 0, 0, w_pad, 0, h_pad))
                reference_labels = F.pad(reference_labels, (0, 0, 0, w_pad, 0, h_pad))

            reference_embeddings = reference_embeddings.view((h + h_pad) // atrous_rate, atrous_rate, 
                                                             (w + w_pad) // atrous_rate, atrous_rate, -1)
            reference_labels = reference_labels.view((h + h_pad) // atrous_rate, atrous_rate, 
                                                             (w + w_pad) // atrous_rate, atrous_rate, -1)
            reference_embeddings = reference_embeddings[:, 0, :, 0, :].contiguous()
            reference_labels = reference_labels[:, 0, :, 0, :].contiguous()
        reference_embeddings_flat = reference_embeddings.view(-1, embedding_dim)
        reference_labels_flat = reference_labels.view(-1, obj_nums)
    else:

        for reference_embeddings, reference_labels, idx in zip(all_reference_embeddings, all_reference_labels, range(ref_num)):
            if atrous_rate > 1:
                h_pad = (atrous_rate - h % atrous_rate) % atrous_rate
                w_pad = (atrous_rate - w % atrous_rate) % atrous_rate
                if h_pad > 0  or w_pad > 0:
                    reference_embeddings = F.pad(reference_embeddings, (0, 0, 0, w_pad, 0, h_pad))
                    reference_labels = F.pad(reference_labels, (0, 0, 0, w_pad, 0, h_pad))

                reference_embeddings = reference_embeddings.view((h + h_pad) // atrous_rate, atrous_rate, 
                                                                 (w + w_pad) // atrous_rate, atrous_rate, -1)
                reference_labels = reference_labels.view((h + h_pad) // atrous_rate, atrous_rate, 
                                                                 (w + w_pad) // atrous_rate, atrous_rate, -1)
                reference_embeddings = reference_embeddings[:, 0, :, 0, :].contiguous()
                reference_labels = reference_labels[:, 0, :, 0, :].contiguous()
        

            reference_embeddings_flat = reference_embeddings.view(-1, embedding_dim)
            reference_labels_flat = reference_labels.view(-1, obj_nums)

            all_reference_embeddings_flat.append(reference_embeddings_flat)
            all_reference_labels_flat.append(reference_labels_flat)

        reference_embeddings_flat = torch.cat(all_reference_embeddings_flat, dim=0)
        reference_labels_flat = torch.cat(all_reference_labels_flat, dim=0)

    
    query_embeddings_flat = query_embeddings.view(-1, embedding_dim)
    
    all_ref_fg = torch.sum(reference_labels_flat, dim=1, keepdim=True) > 0.9
    reference_labels_flat = torch.masked_select(reference_labels_flat, 
        all_ref_fg.expand(-1, obj_nums)).view(-1, obj_nums)
    if reference_labels_flat.size(0) == 0:
        return torch.ones(1, h, w, obj_nums, 1, device=all_ref_fg.device)
    reference_embeddings_flat = torch.masked_select(reference_embeddings_flat, 
        all_ref_fg.expand(-1, embedding_dim)).view(-1, embedding_dim)

    if use_float16:
        query_embeddings_flat = query_embeddings_flat.half()
        reference_embeddings_flat = reference_embeddings_flat.half()
    nn_features = _nearest_neighbor_features_per_object_in_chunks(
        reference_embeddings_flat, query_embeddings_flat, reference_labels_flat, n_chunks)

    nn_features_reshape = nn_features.view(1, h, w, obj_nums, 1)
    nn_features_reshape = (torch.sigmoid(nn_features_reshape + dis_bias.view(1, 1, 1, -1, 1)) - 0.5) * 2

    if ori_size is not None:
        nn_features_reshape = nn_features_reshape.view(h, w, obj_nums, 1).permute(2, 3, 0, 1)
        nn_features_reshape = F.interpolate(nn_features_reshape, size=ori_size, 
            mode='bilinear', align_corners=True).permute(2, 3, 0, 1).view(1, ori_size[0], ori_size[1], obj_nums, 1)

    if use_float16:
        nn_features_reshape = nn_features_reshape.float()
    return nn_features_reshape

########################################################################LOCAL_DIST_MAP
def local_pairwise_distances(
    x, y, max_distance=9, atrous_rate=1, allow_downsample=False):
    """Computes pairwise squared l2 distances using a local search window.
        Use for-loop for saving memory.
    Args:
        x: Float32 tensor of shape [height, width, feature_dim].
        y: Float32 tensor of shape [height, width, feature_dim].
        max_distance: Integer, the maximum distance in pixel coordinates
          per dimension which is considered to be in the search window.
        atrous_rate: Integer, the atrous rate of local matching.
        allow_downsample: Bool, if "True", downsample x and y
          with a stride of 2.
    Returns:
        Float32 distances tensor of shape [height, width, (2 * max_distance + 1) ** 2].
    """
    if allow_downsample:
        ori_height, ori_width, _ = x.size()
        x = x.permute(2, 0, 1).unsqueeze(0)
        y = y.permute(2, 0, 1).unsqueeze(0)
        down_size = (int(ori_height/2) + 1, int(ori_width/2) + 1)
        x = F.interpolate(x, size=down_size, mode='bilinear', align_corners=True)
        y = F.interpolate(y, size=down_size, mode='bilinear', align_corners=True)
        x = x.squeeze(0).permute(1, 2, 0)
        y = y.squeeze(0).permute(1, 2, 0)

    pad_max_distance = max_distance - max_distance % atrous_rate
    padded_y =nn.functional.pad(y, 
        (0, 0, pad_max_distance, pad_max_distance, pad_max_distance, pad_max_distance), 
        mode='constant', value=WRONG_LABEL_PADDING_DISTANCE)

    height, width, _ = x.size()
    dists = []
    for y in range(2 * pad_max_distance // atrous_rate + 1):
        y_start = y * atrous_rate
        y_end = y_start + height
        y_slice = padded_y[y_start:y_end]
        for x in range(2 * max_distance + 1):
            x_start = x * atrous_rate
            x_end = x_start + width
            offset_y = y_slice[:, x_start:x_end]
            dist = torch.sum(torch.pow((x-offset_y),2), dim=2)
            dists.append(dist)
    dists = torch.stack(dists, dim=2)

    return dists

def local_pairwise_distances_parallel(
    x, y, max_distance=9, atrous_rate=1, allow_downsample=True):
    """Computes pairwise squared l2 distances using a local search window.
    Args:
        x: Float32 tensor of shape [height, width, feature_dim].
        y: Float32 tensor of shape [height, width, feature_dim].
        max_distance: Integer, the maximum distance in pixel coordinates
          per dimension which is considered to be in the search window.
        atrous_rate: Integer, the atrous rate of local matching.
        allow_downsample: Bool, if "True", downsample x and y
          with a stride of 2.
    Returns:
        Float32 distances tensor of shape [height, width, (2 * max_distance + 1) ** 2].
    """ 
    ori_height, ori_width, _ = x.size()
    x = x.permute(2, 0, 1).unsqueeze(0)
    y = y.permute(2, 0, 1).unsqueeze(0)
    if allow_downsample:
        down_size = (int(ori_height/2) + 1, int(ori_width/2) + 1)
        x = F.interpolate(x, size=down_size, mode='bilinear', align_corners=True)
        y = F.interpolate(y, size=down_size, mode='bilinear', align_corners=True)

    _, channels, height, width = x.size()

    x2 = x.pow(2).sum(1).view(height, width, 1)

    y2 = y.pow(2).sum(1).view(1, 1, height, width)

    pad_max_distance = max_distance - max_distance % atrous_rate
    
    padded_y = F.pad(y, (pad_max_distance, pad_max_distance, pad_max_distance, pad_max_distance))
    padded_y2 = F.pad(y2, (pad_max_distance, pad_max_distance, pad_max_distance, pad_max_distance), 
        mode='constant', value=WRONG_LABEL_PADDING_DISTANCE)

    offset_y = F.unfold(padded_y, kernel_size=(height, width), 
        stride=(atrous_rate, atrous_rate)).view(channels, height * width, -1).permute(1, 0, 2)
    offset_y2 = F.unfold(padded_y2, kernel_size=(height, width), 
        stride=(atrous_rate, atrous_rate)).view(height, width, -1)
    x = x.view(channels, height * width, -1).permute(1, 2, 0)

    dists = x2 + offset_y2 - 2. * torch.matmul(x, offset_y).view(height, width, -1)
    
    return dists




def local_matching(
    prev_frame_embedding, query_embedding, prev_frame_labels,
    dis_bias=0., multi_local_distance=[15], 
    ori_size=None, atrous_rate=1, use_float16=True, allow_downsample=True, allow_parallel=True):
    """Computes nearest neighbor features while only allowing local matches.
    Args:
        prev_frame_embedding: [height, width, embedding_dim],
          the embedding vectors for the last frame.
        query_embedding: [height, width, embedding_dim],
          the embedding vectors for the query frames.
        prev_frame_labels: [height, width, n_objects], 
        the class labels of the previous frame.
        multi_local_distance: A list of Integer, 
          a list of maximum distance allowed for local matching.
        ori_size: (ori_height, ori_width),
          the original spatial size. If "None", (ori_height, ori_width) = (height, width).
        atrous_rate: Integer, the atrous rate of local matching.
        use_float16: Bool, if "True", use float16 type for matching.
        allow_downsample: Bool, if "True", downsample prev_frame_embedding and query_embedding
          with a stride of 2.
        allow_parallel: Bool, if "True", do matching in a parallel way. If "False", do matching in
          a for-loop way, which will save GPU memory.
    Returns:
        nn_features: A float32 np.array of nearest neighbor features of shape
          [1, height, width, n_objects, 1].
    """
    max_distance = multi_local_distance[-1]

    if ori_size is None:
        height, width = prev_frame_embedding.size()[:2]
        ori_size = (height, width)

    obj_num = prev_frame_labels.size(2)
    pad = torch.ones(1, device=prev_frame_embedding.device) * WRONG_LABEL_PADDING_DISTANCE
    if use_float16:
        query_embedding = query_embedding.half()
        prev_frame_embedding = prev_frame_embedding.half()
        pad = pad.half()

    if allow_parallel:
        d = local_pairwise_distances_parallel(query_embedding, prev_frame_embedding, 
            max_distance=max_distance, atrous_rate=atrous_rate)
    else:
        d = local_pairwise_distances(query_embedding, prev_frame_embedding, 
        max_distance=max_distance, atrous_rate=atrous_rate)
        
    height, width = d.size()[:2]
    
    labels = prev_frame_labels.permute(2, 0, 1).unsqueeze(1)
    labels = F.interpolate(labels, size=(height, width), mode='nearest')

    pad_max_distance = max_distance - max_distance % atrous_rate
    atrous_max_distance = pad_max_distance // atrous_rate

    padded_labels = F.pad(labels,
                        (pad_max_distance, pad_max_distance,
                         pad_max_distance, pad_max_distance,
                         ), mode='constant', value=0)
    offset_masks = F.unfold(padded_labels, kernel_size=(height, width), 
        stride=(atrous_rate, atrous_rate)).view(obj_num, height, width, -1).permute(1, 2, 3, 0) > 0.9
    
    d_tiled = d.unsqueeze(-1).expand((-1,-1,-1,obj_num))  # h, w, num_local_pos, obj_num

    d_masked = torch.where(offset_masks, d_tiled, pad)
    dists, pos = torch.min(d_masked, dim=2)
    multi_dists = [dists.permute(2, 0, 1).unsqueeze(1)]  # n_objects, num_multi_local, h, w
    
    reshaped_d_masked = d_masked.view(height, width, 2 * atrous_max_distance + 1, 
        2 * atrous_max_distance + 1, obj_num)
    for local_dis in multi_local_distance[:-1]:
        local_dis = local_dis // atrous_rate
        start_idx = atrous_max_distance - local_dis
        end_idx = atrous_max_distance + local_dis + 1
        new_d_masked = reshaped_d_masked[:, :, start_idx:end_idx, start_idx:end_idx, :].contiguous()
        new_d_masked = new_d_masked.view(height, width, -1, obj_num)
        new_dists, _ = torch.min(new_d_masked, dim=2)
        new_dists = new_dists.permute(2, 0, 1).unsqueeze(1)
        multi_dists.append(new_dists)

    multi_dists = torch.cat(multi_dists, dim=1)
    multi_dists = (torch.sigmoid(multi_dists + dis_bias.view(-1, 1, 1, 1)) - 0.5) * 2

    if use_float16:
        multi_dists = multi_dists.float()

    ori_height = ori_size[0]
    ori_width = ori_size[1]
    multi_dists = F.interpolate(multi_dists, size=(ori_height, ori_width), 
        mode='bilinear', align_corners=True)
    multi_dists = multi_dists.permute(2, 3, 0, 1)
    multi_dists = multi_dists.view(1, ori_height, ori_width, obj_num, -1)

    return multi_dists