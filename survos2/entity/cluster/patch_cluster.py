import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image
from skimage import img_as_ubyte

from torchvision.ops import roi_align

from transformers import CLIPTokenizerFast, CLIPProcessor, CLIPModel
from tqdm.auto import tqdm

from collections import OrderedDict
from survos2.entity.cluster.cnn_features import CNNFeatures, prepare_3channel
from survos2.entity.cluster.utils import get_surface
from skimage.feature import hog


def calc_sphericity(label_vols):
    sphericities = []
    for i in range(len(label_vols)):
        patch_vol = label_vols[i]
        patch_vol = get_largest_cc(patch_vol)

        res = get_surface(patch_vol, plot3d=False)
        sphericities.append(res[-1])

    return np.sum(np.array(sphericities)) / len(sphericities)


def patch_cluster(
    vec_mat,
    selected_images,
    n_components=32,
    num_clusters=12,
    perplexity=200,
    n_iter=1000,
    skip_px=2,
):
    selected_3channel = prepare_3channel(
        selected_images,
        patch_size=(selected_images[0].shape[0], selected_images[0].shape[1]),
    )
    from entityseg.cluster.cluster_plotting import (
        cluster_scatter,
        plot_clustered_img,
    )
    from entityseg.cluster.clusterer import PatchCluster

    patch_clusterer = PatchCluster(num_clusters, n_components)
    patch_clusterer.prepare_data(vec_mat)
    patch_clusterer.fit()
    patch_clusterer.predict()
    patch_clusterer.embed_TSNE(perplexity=perplexity, n_iter=n_iter)

    print(f"Metrics (DB, Sil, C-H): {patch_clusterer.cluster_metrics()}")
    preds = patch_clusterer.get_predictions()
    twoD_projected_data = patch_clusterer.embedding
    cluster_scatter(twoD_projected_data, preds)

    selected_images_arr = np.array(selected_3channel)
    print(f"Plotting {selected_images_arr.shape} patch images.")
    selected_images_arr = selected_images_arr[:, :, :, 1]

    fig, ax = plt.subplots(figsize=(17, 17))
    plt.title(
        "Windows around a click clustered using deep features and embedded in 2d using T-SNE".format()
    )

    skip_px = skip_px
    plot_clustered_img(
        twoD_projected_data, preds, images=selected_images_arr[:, ::skip_px, ::skip_px]
    )

    return twoD_projected_data, preds


def extract_cnn_features(selected_images, gpu_id=0):
    cnnfeat = CNNFeatures(cuda=True, model="resnet-50", gpu_id=gpu_id)
    num_fv = len(selected_images)
    vec_mat = np.zeros((num_fv, 2048))

    for i, img in enumerate(selected_images):
        fv = []
        img_3channel = np.stack((img, img, img)).T
        fv.extend(cnnfeat.extract_feature(Image.fromarray(img_as_ubyte(img_3channel))))
        vec_mat[i, 0 : len(fv)] = fv

    print(f"Produced image stack of {len(selected_images)}")

    return vec_mat


def extract_hog_features(selected_images, gpu_id=0):
    num_fv = len(selected_images)
    vec_mat = np.zeros((num_fv, 512))

    for i, img in enumerate(selected_images):
        fv = []
        
        fd = hog(
            img, 
            orientations=8,
            pixels_per_cell=(8, 8),
            cells_per_block=(1, 1),
            visualize=False,
        )  
        fv.extend(fd)
        vec_mat[i, 0 : len(fv)] = fv

    print(f"Produced image stack of {len(selected_images)}")

    return vec_mat


def extract_cnn_features2(patch_dict, gpu_id=0):
    cnnfeat = CNNFeatures(cuda=True, model="resnet-50", gpu_id=gpu_id)
    num_fv = len(patch_dict.keys())
    vec_mat = np.zeros((num_fv, 2048))
    roi_num_patches = [len(v) for k, v in patch_dict.items()]
    selected_images = []
    num_imgs = len(patch_dict.keys())

    for i, (k, v) in enumerate(patch_dict.items()):
        fv = []
        idx = np.random.randint(roi_num_patches[i])
        img = v[idx]
        img_3channel = np.stack((img, img, img)).T
        fv.extend(cnnfeat.extract_feature(Image.fromarray(img_as_ubyte(img_3channel))))
        vec_mat[i, 0 : len(fv)] = fv
        selected_images.append(img)

    print(f"Produced image stack of {len(selected_images)}")

    return vec_mat, selected_images



def extract_CLIP_features(selected_images, gpu_id=0, batch_size = 16):
    feature_mat = None
    selected_3channel = [np.stack((img, img, img)).T for img in selected_images]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).to(device)
    image = processor(
    text=None,
    images=selected_3channel[0],
    return_tensors='pt',
    do_rescale=False)['pixel_values'].to(device)
    
    for i in tqdm(range(0, len(selected_3channel), batch_size)):
        batch = selected_3channel[i:i+batch_size]
        batch = processor(
            text=None,
            images=batch,
            return_tensors='pt',
            padding=True
        )['pixel_values'].to(device)
        batch_emb = model.get_image_features(pixel_values=batch)
        batch_emb = batch_emb.squeeze(0)
        batch_emb = batch_emb.cpu().detach().numpy()
        if feature_mat is None:
            feature_mat = batch_emb
        else:
            feature_mat = np.concatenate((feature_mat, batch_emb), axis=0)
    
    return feature_mat

def patch_2d_features(img_volume, bvol_table, gpu_id=0, padvol=10):
    patch_dict, bvol_info = prepare_patch_dict(raligned, raligned_labels, bvol_info)
    print(f"Number of roi extracted {len(patch_dict.keys())}")
    vec_mat = extract_cnn_features(selected_images, gpu_id)
    return vec_mat, selected_images


def patch_2d_features2(img_volume, bvol_table, gpu_id=0, padvol=10):
    # patch_dict, bbs_info = sample_patch_roi(img_vol, bvol_table)
    raligned, raligned_labels, bvol_info = roi_pool_vol(img_volume, [bvol_table], padvol=padvol)
    patch_dict, bvol_info = prepare_patch_dict(raligned, raligned_labels, bvol_info)
    print(f"Number of roi extracted {len(patch_dict.keys())}")
    vec_mat, selected_images = extract_cnn_features(patch_dict, gpu_id)
    return vec_mat, selected_images, patch_dict, bvol_info

def prepare_patch_dict(raligned, raligned_labels, bvol_info):
    patch_dict = OrderedDict()
    for r, rlabel, bv_info in zip(raligned, raligned_labels, bvol_info):
        for patch, lab in zip([l[0, :] for l in r], rlabel):
            if lab in patch_dict:
                patch_dict[lab].append(patch)
            else:
                patch_dict[lab] = [patch]

    return patch_dict, bvol_info


def roi_pool_vol(cropped_vol, filtered_tables, padvol=14):
    roi_aligned = []
    roi_aligned_label = []
    cols = ["bb_s_x", "bb_s_y", "bb_f_x", "bb_f_y", "bb_s_z", "bb_f_z", "class_code"]
    bb2d_sf = np.array(filtered_tables[0][cols])
    print(f"slicing volume of shape {cropped_vol.shape}")

    bbs_info = []

    for jj, z_l in enumerate(range(0, cropped_vol.shape[0] - 1, 16)):
        cropped_slice = cropped_vol[z_l, :].copy()
        cropped_t = torch.FloatTensor(cropped_slice).unsqueeze(0).unsqueeze(0)
        z_u = z_l + 4
        good_bb = []

        for kk, bb in enumerate(bb2d_sf):
            x_s, y_s, x_f, y_f, z_s, z_f, cc = bb
            if (z_l >= z_s) and (z_f >= z_u):
                good_bb.append((bb, kk))
            else:
                pass
                # print(f"rejected {bb}, {z_l}, {z_u}")

        bb2d_centdim = [
            (x_s, y_s, x_f - x_s, y_f - y_s) for (x_s, y_s, x_f, y_f, z_s, z_f, _), _ in good_bb
        ]

        preds = [c for (_, _, _, _, _, _, c), _ in good_bb]
        scores = [1.0 for i in range(len(good_bb))]
        bb_ids = [i for _, i in good_bb]

        bb_info = {
            "bb": bb,
            "bbs": bb2d_centdim,
            "preds": preds,
            "bb_ids": bb_ids,
            "scores": scores,
        }
        padx, pady = padvol, padvol
        expanded_bboxes = np.abs(
            [(0, bb[1] - padx, bb[0] - pady, bb[3] + padx, bb[2] + pady) for bb, _ in good_bb]
        )

        if len(expanded_bboxes) > 0:
            rois = torch.FloatTensor(expanded_bboxes)
            aligned = roi_align(cropped_t, rois, output_size=(64, 64))
            roi_aligned.append(aligned.detach().cpu().numpy())
            roi_aligned_label.append(bb_ids)
            bbs_info.append(bb_info)

    print(len(roi_aligned), len(roi_aligned_label), len(bbs_info))

    return roi_aligned, roi_aligned_label, bbs_info


def roi_pool_vol2(cropped_vol, filtered_tables):
    roi_aligned = []
    print(cropped_vol.shape)
    cols = ["bb_s_x", "bb_s_y", "bb_f_x", "bb_f_y", "bb_s_z", "bb_f_z"]
    bb2d_sf = np.array(filtered_tables[0][cols])

    cropped_t = np.moveaxis(cropped_vol, 0, 1)
    cropped_t = torch.FloatTensor(cropped_t)

    for jj, z_l in enumerate(range(4, cropped_vol.shape[1], 32)):
        z_u = z_l + 10
        print(z_l, z_u)
        good_bb = []

        for bb in bb2d_sf:
            x_s, y_s, x_f, y_f, z_s, z_f = bb

            # print(x_s,y_s,x_f,y_f, z_s,z_f)
            if (z_l >= z_s) and (z_f >= z_u):
                # print(f"bb ok {bb} {z_l} {z_u}")
                good_bb.append(bb)
            else:
                pass
                # print(f"rejected {bb}, {z_l}, {z_u}")

        bb2d_centdim = [
            (x_s, y_s, x_f - x_s, y_f - y_s)
            for (x_s, y_s, x_f, y_f, z_s, z_f) in bb2d_sf
            if (z_l >= z_s) and (z_f >= z_u)
        ]

        preds = ["a" for i in range(bb2d_sf.shape[0])]
        scores = [1.0 for i in range(bb2d_sf.shape[0])]

        pred_info = {"bbs": bb2d_centdim, "preds": preds, "scores": scores}

        plot_bb_2d(cropped_vol[0, (z_l + z_u) // 2, :], pred_info)
        padx, pady = 10, 10

        expanded_bbox = [
            (0, bb[1] - padx, bb[0] - pady, bb[3] + padx, bb[2] + pady)
            for i, bb in enumerate(good_bb)
        ]
        print(expanded_bbox)

        rois = torch.FloatTensor(expanded_bbox)
        aligned = roi_align(cropped_t, rois, output_size=(64, 64))
        roi_aligned.append(aligned.detach().cpu().numpy())

    return roi_aligned


def roi_pool_2d(
    img_orig,
    bbs,
    input_size=(128, 128),
    output_size=(128, 128),
    batch_size=1,
    n_channels=1,
    spatial_scale=1.0,
):
    rois = torch.FloatTensor(bbs)

    x_np = np.arange(batch_size * n_channels * input_size[0] * input_size[1], dtype=np.float32)

    x_np = x_np.reshape((batch_size, n_channels, *input_size))
    np.random.shuffle(x_np)
    x_np = img_orig.reshape((1, 1, img_orig.shape[0], img_orig.shape[1]))
    x = torch.from_numpy(x_np)
    x = x.cuda()
    rois = rois.cuda()
    y = roi_pooling_2d(x, rois, output_size, spatial_scale=spatial_scale)

    return y
