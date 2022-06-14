# Copyright 2022-present NAVER Corp.
# CC BY-NC-SA 4.0
# Available only for non-commercial use

from pdb import set_trace as bb
import os
import torch
import torch.optim as optim
import torchvision.transforms as tvf

from tools import common, trainer
from datasets import *
from core.conv_mixer import ConvMixer
from core.losses import *


def parse_args():
    import argparse
    parser = argparse.ArgumentParser("Script to train PUMP")

    parser.add_argument("--pretrained", type=str, default="", help='pretrained model path')
    parser.add_argument("--save-path", type=str, required=True, help='directory to save model')

    parser.add_argument("--epochs", type=int, default=50, help='number of training epochs')
    parser.add_argument("--batch-size", "--bs", type=int, default=16, help="batch size")
    parser.add_argument("--learning-rate", "--lr", type=str, default=1e-4)
    parser.add_argument("--weight-decay", "--wd", type=float, default=5e-4)
    
    parser.add_argument("--threads", type=int, default=8, help='number of worker threads')
    parser.add_argument("--device", default='cuda')
    
    args = parser.parse_args()
    return args


def main( args ):
    device = args.device
    common.mkdir_for(args.save_path)

    # Create data loader
    db = BalancedCatImagePairs(
            3125, SyntheticImagePairs(RandomWebImages(0,52),distort='RandomTilting(0.5)'),
            4875, SyntheticImagePairs(SfM120k_Images(),distort='RandomTilting(0.5)'),
            8000, SfM120k_Pairs())

    db = FastPairLoader(db,
            crop=256, transform='RandomRotation(20), RandomScale(256,1536,ar=1.3,can_upscale=True), PixelNoise(25)', 
            p_swap=0.5, p_flip=0.5, scale_jitter=0.5)

    print("Training image database =", db)
    data_loader = torch.utils.data.DataLoader(db, batch_size=args.batch_size, shuffle=True, 
            num_workers=args.threads, collate_fn=collate_ordered, pin_memory=False, drop_last=True, 
            worker_init_fn=WorkerWithRngInit())

    # create network
    net = ConvMixer(output_dim=128, hidden_dim=512, depth=7, patch_size=4, kernel_size=9)
    print(f"\n>> Creating {type(net).__name__} net ( Model size: {common.model_size(net)/1e6:.1f}M parameters )")

    # create losses
    loss = MultiLoss(alpha=0.3, 
            loss_sup = PixelAPLoss(nq=20, inner_bw=True, sampler=NghSampler(ngh=7)), 
            loss_unsup = DeepMatchingLoss(eps=0.03))
    
    # create optimizer
    optimizer = optim.Adam( [p for p in net.parameters() if p.requires_grad], 
                            lr=args.learning_rate, weight_decay=args.weight_decay)

    train = MyTrainer(net, loss, optimizer).to(device)

    # initialization
    final_model_path = osp.join(args.save_path,'model.pt')
    last_model_path = osp.join(args.save_path,'model.pt.last')
    if osp.exists( final_model_path ):
        print('Already trained, nothing to do!')
        return
    elif args.pretrained: 
        train.load( args.pretrained )
    elif osp.exists( last_model_path ):
        train.load( last_model_path )

    train = train.to(args.device)
    if ',' in os.environ.get('CUDA_VISIBLE_DEVICES',''):
        train.distribute()

    # Training loop #
    while train.epoch < args.epochs:
        # shuffle dataset (select new pairs)
        data_loader.dataset.set_epoch(train.epoch)

        train(data_loader)

        train.save(last_model_path)

    # save final model
    torch.save(train.model.state_dict(), open(final_model_path,'wb'))


totensor = tvf.Compose([
    common.ToTensor(), 
    tvf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

class MyTrainer (trainer.Trainer):
    """ This class implements the network training.
        Below is the function I need to overload to explain how to do the backprop.
    """
    def forward_backward(self, inputs):
        assert torch.is_grad_enabled() and self.net.training

        (img1, img2), labels = inputs
        output1 = self.net(totensor(img1))
        output2 = self.net(totensor(img2))

        loss, details = trainer.get_loss(self.loss(output1, output2, img1=img1, img2=img2, **labels))
        trainer.backward(loss)
        return details



if __name__ == '__main__':
    main(parse_args())
