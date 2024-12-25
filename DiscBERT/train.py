import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from pprint import pprint
from dataset.data_module import DataModule
from lightning.pytorch import seed_everything
import lightning.pytorch as pl
import torch
import argparse
from models.DiscBERT import DiscBERT

def train(args):
    dm = DataModule(args)

    trainer = pl.Trainer(
        devices=args.devices,
        strategy=args.strategy,
        accelerator=args.accelerator,
        precision=args.precision,
        val_check_interval = args.val_check_interval,
        max_epochs = args.max_epochs,
    )

    model = DiscBERT(args)
    if args.predict:
        trainer.predict(model, datamodule=dm)
    else:
        if args.test:
            trainer.test(model, datamodule=dm)
        elif args.validate:
            trainer.validate(model, datamodule=dm)
        else:
            trainer.fit(model , datamodule=dm)


def main():
    parser = argparse.ArgumentParser(description="hallGPT")
    parser.add_argument('--test', action='store_true', help="only run test set")
    parser.add_argument('--validate', action='store_true', help="only run validation set")
    parser.add_argument('--train', action='store_true', help="only run validation set")
    parser.add_argument('--predict', action='store_true', help="only run validation set")
    parser.add_argument('--trainroad', type=str, default='./data/train.csv', help="road-train data")
    parser.add_argument('--valroad', type=str, default='./data/val.csv', help="road-val data")
    parser.add_argument('--testroad', type=str, default='./data/manual_annotaion.csv', help="road-test data")
    parser.add_argument('--predictroad', type=str, default='./pred_f.csv', help="road-predict data")
    parser.add_argument('--base_dir', type=str, default=r'./data/mimic_cxr/images', help="base dir to help find images")
    parser.add_argument('--batch_size', default=8, type=int, help="use for training duration per worker")
    parser.add_argument('--val_batch_size', default=16, type=int, help="use for validation duration per worker")
    parser.add_argument('--test_batch_size', default=16, type=int, help="use for testing duration per worker")
    parser.add_argument('--prefetch_factor', default=4, type=int, help="use for training duration per worker")
    parser.add_argument('--num_workers', default=8, type=int, help="Cpu num for dataloaders")
    parser.add_argument('--val_check_interval', type=float, default=1.0, help='How often to check the validation set')
    parser.add_argument('--max_epochs', type=int, default=15, help='Stop training once this number of epochs is reached')
    parser.add_argument('--devices', type=int, default=1, help='how many gpus to use')
    parser.add_argument('--strategy', type=str, default='ddp', help='default ddp for multi-gpus')
    parser.add_argument('--accelerator', type=str, default="gpu", choices=["cpu", "gpu", "tpu", "ipu", "hpu", "mps"],help='accelerator types')
    parser.add_argument('--precision', type=str, default='32',help='16 or 32 bf16-mixed, using for original pytorch amp auto cast')
    parser.add_argument('--savedmodel_path', type=str, default='save/mimic_11_08_0_1k')
    parser.add_argument('--delta_file', type=str, default=None, help='the delta file to load')
    parser.add_argument('--learning_rate', default=2e-5, type=float, help='initial learning rate')
    parser.add_argument('--BertModel', default="./bert-base-uncased", type=str, help='BertModel path')

    parser.set_defaults(predict=True)
    parser.set_defaults(delta_file='./discbert.pth')
    args = parser.parse_args()
    os.makedirs(args.savedmodel_path, exist_ok=True)
    pprint(vars(args))
    seed_everything(42, workers=True)
    train(args)

if __name__ == '__main__':
    main()
    print('success')



