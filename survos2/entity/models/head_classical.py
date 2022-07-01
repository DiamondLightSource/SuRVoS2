import os
import h5py
from PIL import Image
from skimage import img_as_ubyte

import numpy as np
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier,
)

from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from survos2.entity.utils import get_surface, pad_vol
from survos2.entity.entities import offset_points, make_entity_df
from survos2.frontend.nb_utils import show_images, slice_plot
from survos2.server.features import features_factory, generate_features
from survos2.server.filtering import ndimage_laplacian, gaussian_blur_kornia, spatial_gradient_3d
from survos2.server.state import cfg
from survos2.server.filtering.morph import dilate, erode, median
from survos2.entity.pipeline_ops import make_features

from survos2.entity.cluster.cnn_features import CNNFeatures, prepare_3channel
from survos2.entity.instance.detector import prepare_patch_dataloaders_and_entities
from survos2.entity.instance.detector import prepare_filtered_patch_dataset
from survos2.entity.models.head_cnn import setup_fpn_for_extraction, process_fpn3d_pred


def classical_head_train(features, labels, saved_cls=False, n_components=7):
    print(f"Training on features of shape {features.shape}")

    reduced_data= make_pipeline(StandardScaler(), PCA(n_components=n_components)).fit_transform(np.nan_to_num(features))

    #reduced_data = PCA(n_components=n_components).fit_transform(np.nan_to_num(features))
    # params = {'n_neighbors':20,
    #         'min_dist':0.3,
    #         'n_components':2,
    #         'metric':'euclidean'}
    # reduced_data = UMAP(**params).fit_transform(np.nan_to_num(features))
    
    X_train, X_test, y_train, y_test = train_test_split(
        reduced_data, labels, test_size=0.9, random_state=41
    )
    n_classes = len(np.unique(y_train))
    print(f"Number of classes: {n_classes}")
    n_estimators = 100

    gbc_params = {
        "n_estimators": 100,
        "max_leaf_nodes": 4,
        "max_depth": None,
        "random_state": 2,
        "min_samples_split": 5,
        "learning_rate": 0.1,
        "subsample": 0.5,
    }

    # rfc = RandomForestClassifier(n_estimators=n_estimators, max_depth=6)
    etc = ExtraTreesClassifier(n_estimators=n_estimators)
    # gbc = GradientBoostingClassifier(**gbc_params)
    #mlp = MLPClassifier(random_state=1, max_iter=300)

    #svc = svm.SVC(kernel="linear", C=1)

    X = X_train
    y = y_train

    # Standardize
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X_std = (X - mean) / std
    X_test_std = (X_test - mean) / std

    # rfc.fit(X, y)
    etc.fit(X, y)
    # gbc.fit(X, y)
    # svc.fit(X, y)
    #mlp.fit(X, y)

    trained_classifiers = {}
    # scores = rfc.score(X, y)
    # preds = rfc.predict(X_test)
    # trained_classifiers = {"rfc": {"score": accuracy_score(y_test, preds)}}
    # trained_classifiers["rfc"]["classifier"] = rfc
    # print(f"Random forest accuracy score: {accuracy_score(y_test, preds)}")

    # # print(scores_svm)
    scores_etc = etc.score(X, y)
    preds_etc = etc.predict(X_test)
    trained_classifiers["etc"] = {}
    trained_classifiers["etc"]["score"] = accuracy_score(y_test, preds_etc)
    trained_classifiers["etc"]["classifier"] = etc
    print(f"Extra random tree accuracy score: {accuracy_score(y_test, preds_etc)}")

    # preds_gbc = gbc.predict(X_test)
    # trained_classifiers["gbc"] = {}
    # trained_classifiers["gbc"]["score"] = accuracy_score(y_test, preds_gbc)
    # trained_classifiers["gbc"]["classifier"] = gbc
    # print(
    #     f"Gradient boosting classifier accuracy score: {accuracy_score(y_test, preds_gbc)}"
    # )

    # preds_svc = svc.predict(X_test_std)
    # trained_classifiers["svc"] = {}
    # trained_classifiers["svc"]["score"] = accuracy_score(y_test, preds_gbc)
    # trained_classifiers["svc"]["classifier"] = svc
    # print(
    #     f"Gradient boosting classifier accuracy score: {accuracy_score(y_test, preds_gbc)}"
    # )

    # preds_mlp = mlp.predict(X_test)
    # trained_classifiers["mlp"] = {}
    # trained_classifiers["mlp"]["score"] = accuracy_score(y_test, preds_mlp)
    # trained_classifiers["mlp"]["classifier"] = mlp
    # print(
    #     f"MLP classifier accuracy score: {accuracy_score(y_test, preds_mlp)}"
    # )

    return trained_classifiers


def classical_head_validate(features, labels, classifier, n_components):
    reduced_data= make_pipeline(StandardScaler(), PCA(n_components=n_components)).fit_transform(np.nan_to_num(features))
    #reduced_data = PCA(n_components=n_components).fit_transform(np.nan_to_num(features))
    # params = {'n_neighbors':20,
    #         'min_dist':0.3,
    #         'n_components':2,
    #         'metric':'euclidean'}
    # reduced_data = UMAP(**params).fit_transform(np.nan_to_num(features))

    print("Predicting...")
    preds = classifier.predict(reduced_data)
    proba = classifier.predict_proba(reduced_data)
    acc_score = accuracy_score(preds, labels)
    print(preds)
    print(acc_score)
    return preds, proba


def generate_feature_vols(padded_vol, padded_proposal, wf):
    feature_params = [
        [gaussian_blur_kornia, {"sigma": 3}],
        [ndimage_laplacian, {"kernel_size": 4}],
        #[spatial_gradient_3d, {}],
        #[median, {"median_size" : 6, "num_iter": 2}]
    ]

    roi_crop = [
        0,
        padded_vol.shape[0],
        0,
        padded_vol.shape[1],
        0,
        padded_vol.shape[2],
    ]

    features = generate_features(padded_vol, feature_params, roi_crop, 1.0)
    print(f"Generated {len(features.filtered_layers)} features.")
    return features



def prepare_classical_features(
    wf,
    filtered_vols,
    entities,
    model_file=None,
    bvol_dim=(32, 32, 32),
    stage_train=True,
    plot_all=False,
    flip_xy=False,
):
    if stage_train:
        fvd, targs_all = prepare_filtered_patch_dataset(
            entities, filtered_vols.filtered_layers, bvol_dim=bvol_dim, flip_xy=False
        )
    else:
        entities = np.array(make_entity_df(entities, flipxy=True))
        fvd, targs_all = prepare_filtered_patch_dataset(
            entities, filtered_vols.filtered_layers, bvol_dim=bvol_dim, flip_xy=False, offset_type="None"
        )
    print(f"Prepared classical feature volumes of shape {fvd[0][0][0].shape}")
    features = extract_classical_features(fvd, bvol_dim, plot_all=plot_all, resnet=True )
    print(f"Extracted classical features of shape {features.shape}")
    if model_file:
        model3d = setup_fpn_for_extraction(wf, model_file)
        fpn_features = process_fpn3d_pred(model3d, fvd)
        print(f"Extracted fpn features of shape {fpn_features.shape}")

        features = np.hstack((features, fpn_features))

    return features, fvd



def get_resnet_feature(img, model="resnet-50", gpu_id=0):
    cnnfeat = CNNFeatures(cuda=True, model=model, gpu_id=gpu_id)
    fv = []
    img_3channel = np.stack((img, img, img)).T
    fv.extend(cnnfeat.extract_feature(Image.fromarray(img_as_ubyte(img_3channel))))
    return fv

def extract_classical_features(fvd, bvol_dim, plot_all=False, resnet=False):
    features = []
    for i in range(len(fvd)):
        curvols, target = fvd[i]
        segvol = curvols[-1]
        
        label = target["labels"]
        max_slice, slice_incr = segvol.shape[0], segvol.shape[0] // 3

        if plot_all:
            show_images(
                [curvols[0][i, :] for i in range(0, max_slice, slice_incr)],
                [label for i in range(0, max_slice, slice_incr)],
                figsize=(1, 1),
            )
            show_images(
                [segvol[i, :] for i in range(0, max_slice, slice_incr)],
                [label for i in range(0, max_slice, slice_incr)],
                figsize=(1, 1),
            )

        if np.sum(segvol) > 0:
            sphericity = get_surface((segvol > 0) * 1.0)[-1]
        else:
            sphericity = 0

        from survos2.entity.utils import get_largest_cc
        size_largest_cc = np.sum(get_largest_cc((segvol > 0) * 1.0)) / (segvol.shape[0] * segvol.shape[1] * segvol.shape[2])
        

        fv = [
                [
                np.sum(v[segvol > 0]) / (segvol.shape[0] * segvol.shape[1] * segvol.shape[2]),
                np.mean(v[segvol > 0]),
                np.std(v[segvol > 0]),
                np.median(v[segvol > 0]),
                np.mean(v),
                np.std(v),
                np.median(v),
                np.mean(v[16:48,16:48,16:48]),
                np.std(v[16:48,16:48,16:48]),
                sphericity,
                size_largest_cc]
            
            for v in curvols
        ]

        fv = np.array(fv).flatten()
        fv = list(fv)

        if resnet:
            resnet_fv = get_resnet_feature(curvols[2][bvol_dim[0]//2,:])
            fv.extend(resnet_fv)
        
        fv = np.array(fv).flatten()
        fv = np.nan_to_num(fv)
        features.append(fv)
        
    print(fv.shape[0])
    #print(f" Features Length: {len(features[0][0])} Curvols length: {len(curvols)}")
    features = np.array(features).reshape((len(fvd), fv.shape[0]))

    print(features)
    return features


def trainvalidate_classical_head(
    wf,
    filtered_vols,
    gt_train_entities,
    gt_val_entities,
    model_file=None,
    n_components=7,
    bvol_dim=(32, 32, 32),
    flip_xy=False,
    plot_all=False,
):
    features_train, filtered_patch_dataset_train = prepare_classical_features(
        wf,
        filtered_vols,
        gt_train_entities,
        model_file,
        bvol_dim=bvol_dim,
        plot_all=plot_all,
        flip_xy=flip_xy,
    )
    print(f"Prepared features of shape {features_train.shape}")
    features_val, filtered_patch_dataset_val = prepare_classical_features(
        wf,
        filtered_vols,
        gt_val_entities,
        model_file,
        bvol_dim=bvol_dim,
        plot_all=plot_all,
        flip_xy=flip_xy,
    )
    print(f"Prepared validation features of shape {features_val.shape}")

    trained_classifiers = classical_head_train(
        features_train, filtered_patch_dataset_train.labels, n_components=n_components
    )

    preds, proba = classical_head_validate(
        features_val,
        filtered_patch_dataset_val.labels,
        trained_classifiers["etc"]["classifier"],
        n_components=n_components,
    )

    return (
        trained_classifiers,
        filtered_patch_dataset_train,
        filtered_patch_dataset_val,
        [features_train, features_val],
    )


def filter_scores(proba, score_thresh=0.8):
    probmax = []
    maxarg = np.argmax(proba, axis=1)
    for i, p in enumerate(proba):
        probmax.append(p[maxarg[i]])
    probmax = np.array(probmax)
    detected = probmax > score_thresh
    return detected



def classical_detect(
    wf,
    filtered_vols,
    classifier,
    class_proposal_fname,
    model_file=None,
    padding=(32, 32, 32),
    score_thresh=0.5,
    area_min=50,
    area_max=3000000,
    n_components=6,
    plot_all=False,
    flip_xy=False,
):
    proposal_segmentation_fullpath = os.path.join(
        wf.params["outdir"], class_proposal_fname
    )
    from survos2.entity.instance.detector import prepare_component_table

    component_table, padded_proposal = prepare_component_table(
        wf,
        proposal_segmentation_fullpath,
        area_min=area_min,
        area_max=area_max,
        padding=padding,
    )
    proposal_entities = np.array(component_table[["z", "x", "y", "class_code"]])
    print(f"Produced proposal entities of shape {proposal_entities.shape}")

    detected, preds, proba, fvd = classical_prediction(
        wf,
        filtered_vols,
        classifier,
        proposal_entities,
        model_file,
        n_components=n_components,
        score_thresh=score_thresh,
        bvol_dim=np.array(padding) // 2,
        plot_all=plot_all,
        flip_xy=flip_xy,
    )
    print(
        f"ran_classical_head generated detections of shape {detected.shape} and preds {preds.shape}"
    )
    proposal_entities[:, 3] = preds
    detected_entities = proposal_entities[detected]

    offset_detected_entities = offset_points(detected_entities, np.array(padding) // 2)
    slice_plot(
        wf.vols[0],
        offset_detected_entities,
        None,
        (wf.vols[0].shape[0] // 2, wf.vols[0].shape[1] // 2, wf.vols[0].shape[2] // 2),
    )
    return offset_detected_entities, proba, fvd


def prepare_classical_detector_fvols(
    wf, class_proposal_fname, padding, area_min=10000, area_max=1000000
):
    proposal_segmentation_fullpath = os.path.join(
        wf.params["outdir"], class_proposal_fname
    )
    padded_vol = pad_vol(wf.vols[1], padding)
    padded_mask = pad_vol(wf.bg_mask, padding)
    _, padded_proposal = prepare_component_table(
        wf, proposal_segmentation_fullpath, area_min=area_min, area_max=area_max
    )
    feature_vols = generate_feature_vols(padded_vol, padded_proposal, wf)
    return prepare_feature_vols(feature_vols, padded_vol, padded_proposal)


def classical_detector_predict(
    wf,
    fvol,
    proposal_entities,
    classifier,
    score_thresh=0.75,
    bvol_dim=(32, 32, 32),
    n_components=6,
):

    detected, preds = classical_prediction(
        wf,
        fvol,
        classifier,
        proposal_entities,
        n_components=n_components,
        score_thresh=score_thresh,
        bvol_dim=bvol_dim,
    )
    print(
        f"ran_classical_head generated detections of shape {detected.shape} and preds {preds.shape}"
    )
    proposal_entities[:, 3] = preds
    return proposal_entities[detected]



def prepare_feature_vols(
    features, padded_vol, padded_proposal, additional_feature_vols=None
):
    feature_list = [padded_vol, padded_proposal]
    if additional_feature_vols is not None:
        for additional_feature in additional_feature_vols:
            feature_list.append(additional_feature)
            print(f"Using additional feature of shape {additional_feature.shape}")
    print(
        f"Preparing feature list with padded vol and padded proposal of shape {padded_vol.shape} {padded_proposal.shape} "
    )
    features_stack = features_factory(feature_list)
    return features_stack


def prepare_classical_detector_data2(
    main_vol,
    gt_entities,
    proposal,
    padding,
    additional_feature_vols=None,
    area_min=0,
    area_max=1e14,
    flip_xy=False,
):

    # component_table, padded_proposal = prepare_component_table(
    #    wf, proposal_fullpath, area_min=area_min, area_max=area_max, padding=padding
    # )
    padding = np.array(padding)
    proposal_filt = proposal
    padded_proposal = pad_vol(proposal_filt, padding // 2)
    padded_vol = pad_vol(main_vol, np.array(padding) // 2)

    padded_additional_feature_vols = []
    if additional_feature_vols is not None:
        for additional_feature in additional_feature_vols:
            padded_additional_feature = pad_vol(additional_feature, np.array(padding) // 2)
            padded_additional_feature_vols.append(padded_additional_feature)
    else:
        padded_additional_feature_vols = None

    print(f"Padded vol {padded_vol.shape} and padded_proposal {padded_proposal.shape}")
    #feature_vols = generate_feature_vols(padded_vol, padded_proposal, wf)
    feature_vols = []
    fvol = prepare_feature_vols(
        feature_vols,
        padded_vol,
        padded_proposal,
        additional_feature_vols=padded_additional_feature_vols,
    )

    (
        dataloaders,
        gt_train_entities,
        gt_val_entities,
    ) = prepare_patch_dataloaders_and_entities(
        main_vol, gt_entities,  padding=padding, flip_xy=flip_xy
    )

    return gt_train_entities, gt_val_entities, fvol


def prepare_classical_detector_data(
    wf,
    gt_entities,
    proposal_fullpath,
    padding,
    additional_feature=None,
    area_min=0,
    area_max=1e14,
    flip_xy=False,
):

    # component_table, padded_proposal = prepare_component_table(
    #    wf, proposal_fullpath, area_min=area_min, area_max=area_max, padding=padding
    # )
    padding = np.array(padding)
    proposal_filt = load_and_prepare_proposal(proposal_fullpath)
    padded_proposal = pad_vol(proposal_filt, padding // 2)
    padded_vol = pad_vol(wf.vols[0], np.array(padding) // 2)

    if additional_feature is not None:
        padded_additional_feature = pad_vol(additional_feature, np.array(padding) // 2)
    else:
        padded_additional_feature = None

    print(f"Padded vol {padded_vol.shape} and padded_proposal {padded_proposal.shape}")
    #feature_vols = generate_feature_vols(padded_vol, padded_proposal, wf)
    feature_vols = []
    fvol = prepare_feature_vols(
        feature_vols,
        padded_vol,
        padded_proposal,
        additional_feature=padded_additional_feature,
    )

    (
        dataloaders,
        gt_train_entities,
        gt_val_entities,
    ) = prepare_patch_dataloaders_and_entities(
        wf, gt_entities, proposal_fullpath, padding=padding, flip_xy=flip_xy
    )

    return gt_train_entities, gt_val_entities, fvol


def load_and_prepare_proposal(map_fullpath):
    print(map_fullpath)
    with h5py.File(map_fullpath, "r") as hf:
        print(hf.keys())
        proposal = hf["map"][:]
    cfg["feature_params"] = {}
    from survos2.entity.pipeline import Patch

    p = Patch({"Main": proposal}, {}, {}, {})
    # cfg["feature_params"]["out2"] = [[erode, {"thresh": 0.5, "num_iter": 1}]]
    cfg["feature_params"]["out2"] = [[dilate, {"thresh": 0.5, "num_iter": 1}]]
    p = make_features(p, cfg)
    return p.image_layers["out2"]


def classical_prediction(
    wf,
    filtered_vols,
    classifier,
    entities,
    model_file=None,
    bvol_dim=(32, 32, 32),
    n_components=7,
    score_thresh=0.5,
    plot_all=False,
    flip_xy=False,
    standardize=False,
    offset=False
):


    if offset:
        prepared_entities = offset_points(entities, - 2 *np.array(bvol_dim))
        print("Offset points for prediction.")
    else:
        prepared_entities = entities
        print("No prediction offset")
    print(f"Running classical feature based detector with bvol_dim: {bvol_dim}")
    
    features, fvd_ = prepare_classical_features(
        wf,
        filtered_vols,
        prepared_entities,
        model_file,
        bvol_dim=bvol_dim,
        plot_all=plot_all,
        flip_xy=flip_xy,
        stage_train=False
    )

    if standardize:
        mean = features.mean(axis=0)
        std = features.std(axis=0)
        features = (features - mean) / std

    print(features)
    features = np.nan_to_num(features)
    reduced_data= make_pipeline(StandardScaler(), PCA(n_components=n_components)).fit_transform(np.nan_to_num(features))

    #reduced_data = PCA(n_components=n_components).fit_transform(features)
    # params = {'n_neighbors':20,
    #         'min_dist':0.3,
    #         'n_components':2,
    #         'metric':'euclidean'}
    # reduced_data = UMAP(**params).fit_transform(np.nan_to_num(features))
    print("Predicting")
    #reduced_data = features
    preds = classifier.predict(reduced_data)
    proba = classifier.predict_proba(reduced_data)
    detected = filter_scores(proba, score_thresh=score_thresh)
  
    return detected, preds, proba, fvd_


def make_classical_detector_prediction2(
    wf,
    filtered_vols,
    classifier,
    class_proposal_fname,
    model_file=None,
    padding=(32, 32, 32),
    score_thresh=0.5,
    area_min=50,
    area_max=3000000,
    n_components=6,
    plot_all=False,
    flip_xy=False,
):
    proposal_segmentation_fullpath = os.path.join(
        wf.params["outdir"], class_proposal_fname
    )
    from survos2.entity.instance.detector import prepare_component_table
    component_table, padded_proposal = prepare_component_table(
        wf, proposal_segmentation_fullpath, area_min=area_min, area_max=area_max, padding=padding
    )
    proposal_entities = np.array(component_table[["z", "x", "y", "class_code"]])
    print(f"Produced proposal entities of shape {proposal_entities.shape}")
    
    detected, preds, proba, fvd = run_classical_head(
        wf,
        filtered_vols,
        classifier,
        proposal_entities,
        model_file,
        n_components=n_components,
        score_thresh=score_thresh,
        bvol_dim=np.array(padding) //2,
        plot_all=plot_all,
        flip_xy=flip_xy
    )
    print(
        f"ran_classical_head generated detections of shape {detected.shape} and preds {preds.shape}"
    )
    proposal_entities[:, 3] = preds
    detected_entities = proposal_entities[detected]

    
    #print(detected_entities, detected_entities.shape)
    #offset_detected_entities = detected_entities
    offset_detected_entities = offset_points(detected_entities, np.array(padding) //2)
    slice_plot(
        wf.vols[0],
        offset_detected_entities,
        None,
        (wf.vols[0].shape[0] // 2, wf.vols[0].shape[1] // 2, wf.vols[0].shape[2] // 2),
    )
    return offset_detected_entities, proba, fvd



