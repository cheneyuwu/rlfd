from config import parse_args
from data_loader import get_data_loader

# from models.gan import GAN
# from models.dcgan import DCGAN_MODEL
# from models.wgan_clipping import WGAN_CP
# from models.wgan_gradient_penalty import WGAN_GP

from td3fd.td3.shaping import ImgGANShaping

from torchvision import utils

import numpy as np


def get_infinite_batches(data_loader):
    while True:
        for i, (images, _) in enumerate(data_loader):
            yield images


def main(args):
    # model = None
    # if args.model == "GAN":
    #     model = GAN(args)
    # elif args.model == "DCGAN":
    #     model = DCGAN_MODEL(args)
    # elif args.model == "WGAN-CP":
    #     model = WGAN_CP(args)
    # elif args.model == "WGAN-GP":
    #     model = WGAN_GP(args)
    # else:
    #     print("Model type non-existing. Try again.")
    #     exit(-1)
    model = ImgGANShaping(
        dims={"o": (3, 32, 32), "g": (0,), "u": (0,)},
        max_u=1.0,
        gamma=0.99,
        layer_sizes=[256, 256],
        potential_weight=1.0,
        norm_obs=True,
        norm_eps=0.01,
        norm_clip=5,
    )

    # Load datasets to train and test loaders
    train_loader, test_loader = get_data_loader(args)
    # feature_extraction = FeatureExtractionTest(train_loader, test_loader, args.cuda, args.batch_size)

    # # Start model training
    # if args.is_train == "True":
    #     model.train(train_loader)

    # # start evaluating on test data
    # else:
    #     model.evaluate(test_loader, args.load_D, args.load_G)
    #     # for i in range(50):
    #     #    model.generate_latent_walk(i)

    train_loader = get_infinite_batches(train_loader)
    for i in range(args.generator_iters):
        images = train_loader.__next__()
        dloss, gloss = model.train(
            {"o": images, "g": np.empty((args.batch_size, 0)), "u": np.empty((args.batch_size, 0))}
        )
        if i % 100 == 0:
            model.evaluate()
            # grid = utils.make_grid(images)
            # print("Grid of 8x8 images saved to 'dgan_model_image.png'.")
            # utils.save_image(grid, "dgan_model_image.png")
            # print(dloss)


if __name__ == "__main__":
    """
    python test_gan_img.py --model WGAN-GP \
               --is_train True \
               --download True \
               --dataroot datasets/cifar \
               --dataset cifar \
               --generator_iters 40000 \
               --cuda True \
               --batch_size 64
    """
    args = parse_args()
    main(args)
