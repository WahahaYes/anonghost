import argparse
import glob
import os
import random
import time

import cv2
import numpy as np
import torch

from arcface_model.iresnet import iresnet100
from coordinate_reg.image_infer import Handler
from insightface_func.face_detect_crop_multi import Face_detect_crop
from models.config_sr import TestOptions
from models.pix2pix_model import Pix2PixModel
from network.AEI_Net import AEI_Net
from utils.inference.core import model_inference
from utils.inference.image_processing import crop_face
from utils.inference.video_processing import (
    face_enhancement,
    get_final_images,
    get_target,
)


def shuffle_no_stay_same_place(lst):
    """
    Shuffles a list and ensures that no elements remain in the same place.

    Args:
        lst (list): The list to be shuffled.

    Returns:
        list: The shuffled list.
    """
    if len(lst) <= 1:
        return lst

    # Create a copy of the list to avoid modifying the original list
    lst_copy = lst[:]

    # Shuffle the list using random.shuffle
    random.shuffle(lst_copy)

    # While any element is still in its original position, reshuffle
    while any(np.array_equal(a, b) for a, b in zip(lst, lst_copy)):
        random.shuffle(lst_copy)

    return lst_copy


def init_models(args):
    # model for face cropping
    app = Face_detect_crop(name="antelope", root="./insightface_func/models")
    app.prepare(ctx_id=0, det_thresh=0.6, det_size=(640, 640))

    # main model for generation
    G = AEI_Net(args.backbone, num_blocks=args.num_blocks, c_id=512)
    G.eval()
    G.load_state_dict(torch.load(args.G_path, map_location=torch.device("cpu")))
    G = G.cuda()
    G = G.half()

    # arcface model to get face embedding
    netArc = iresnet100(fp16=False)
    netArc.load_state_dict(torch.load("arcface_model/backbone.pth"))
    netArc = netArc.cuda()
    netArc.eval()

    # model to get face landmarks
    handler = Handler("./coordinate_reg/model/2d106det", 0, ctx_id=0, det_size=640)

    # model to make superres of face, set use_sr=True if you want to use super resolution or use_sr=False if you don't
    if args.use_sr:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        torch.backends.cudnn.benchmark = True
        opt = TestOptions()
        # opt.which_epoch ='10_7'
        model = Pix2PixModel(opt)
        model.netG.train()
    else:
        model = None

    return app, G, netArc, handler, model


def main(args):
    app, G, netArc, handler, model = init_models(args)

    # get crops from source images
    print("List of source paths: ", args.source_paths)
    source = []
    try:
        for source_path in args.source_paths:
            img = cv2.imread(source_path)
            img = crop_face(img, app, args.crop_size)[0]
            source.append(img[:, :, ::-1])
    except TypeError:
        print("Bad source images!")
        exit()

    # get full frames from video
    full_frames = []
    for frame in glob.glob(f"{args.target_dir}/*.png"):
        full_frames.append(cv2.imread(frame))

    # get target faces that are used for swap
    set_target = True
    print("List of target paths: ", args.target_faces_paths)
    if not args.target_faces_paths:
        target = get_target(full_frames, app, args.crop_size)
        set_target = False
    else:
        target = []
        try:
            for target_faces_path in args.target_faces_paths:
                img = cv2.imread(target_faces_path)
                img = crop_face(img, app, args.crop_size)[0]
                target.append(img)

            if len(target) > 1 and args.should_shuffle_target_faces:
                target = shuffle_no_stay_same_place(target)
        except TypeError:
            print("Bad target images!")
            exit()

    start = time.time()

    for s, t in zip(source, target):
        final_frames_list, crop_frames_list, full_frames, tfm_array_list = (
            model_inference(
                full_frames,
                [s],
                [t],
                netArc,
                G,
                app,
                set_target,
                similarity_th=args.similarity_th,
                crop_size=args.crop_size,
                BS=args.batch_size,
            )
        )
        if args.use_sr:
            final_frames_list = face_enhancement(final_frames_list, model)

        full_frames = get_final_images(
            final_frames_list,
            crop_frames_list,
            full_frames,
            tfm_array_list,
            handler,
        )

    # replace the images
    for i, frame in enumerate(glob.glob(f"{args.target_dir}/*.png")):
        cv2.imwrite(frame, full_frames[i])

    print("Total time: ", time.time() - start)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Generator params
    parser.add_argument(
        "--G_path",
        default="weights/G_unet_2blocks.pth",
        type=str,
        help="Path to weights for G",
    )
    parser.add_argument(
        "--backbone",
        default="unet",
        const="unet",
        nargs="?",
        choices=["unet", "linknet", "resnet"],
        help="Backbone for attribute encoder",
    )
    parser.add_argument(
        "--num_blocks", default=2, type=int, help="Numbers of AddBlocks at AddResblock"
    )

    parser.add_argument("--batch_size", default=40, type=int)
    parser.add_argument("--crop_size", default=224, type=int, help="Don't change this")
    parser.add_argument(
        "--use_sr",
        default=True,
        type=bool,
        help="True for super resolution on swap images",
    )
    parser.add_argument(
        "--similarity_th",
        default=0.15,
        type=float,
        help="Threshold for selecting a face similar to the target",
    )

    parser.add_argument(
        "--source_paths",
        default=["examples/images/mark.jpg", "examples/images/elon_musk.jpg"],
        nargs="+",
    )
    parser.add_argument(
        "--target_faces_paths",
        default=[],
        nargs="+",
        help="It's necessary to set the face/faces in the video to which the source face/faces is swapped. You can skip this parametr, and then any face is selected in the target video for swap.",
    )
    parser.add_argument(
        "--should_shuffle_target_faces",
        default=False,
        type=bool,
        help="Shuffles the targets",
    )

    # parameters for image to video
    parser.add_argument(
        "--target_dir",
        type=str,
        help="It's necessary for image to video swap",
    )

    args = parser.parse_args()
    main(args)
