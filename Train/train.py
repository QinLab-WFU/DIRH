import json
import os
import time
# from Noise import add_noise_to_labels

import torch
import torch.nn.functional as F
from loguru import logger
from timm.utils import AverageMeter

from Listwise.loss import ListwiseLoss
from _data import build_loaders, get_topk, get_class_num
from _network import build_model
from _utils import (
    build_optimizer,
    calc_learnable_params,
    calc_map_eval,
    EarlyStopping,
    init,
    print_in_md,
    save_checkpoint,
    seed_everything,
    validate_smart,
    rename_output,
)
from config import get_config


def train_epoch(args, dataloader, net, criterion, optimizer, epoch):
    tic = time.time()

    stat_meters = {}
    for x in ["t_loss", "s_loss", "loss", "mAP"]:
        stat_meters[x] = AverageMeter()

    net.train()
    for images, labels, _ in dataloader:
        images, labels = images.to(args.device), labels.to(args.device)

        embeddings = net(images)

        embeddings = F.normalize(embeddings)
        sim_mat = embeddings @ embeddings.T

        s_loss = criterion.fwd_s_ndcg(
            sim_mat,
            labels,
            tau=args.tau,
            relevance_metric=args.relevance_metric
        )
        stat_meters["s_loss"].update(s_loss)

        loss = s_loss
        stat_meters["loss"].update(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # to check overfitting
        map_v = calc_map_eval(embeddings.sign(), labels)
        stat_meters["mAP"].update(map_v)

        torch.cuda.empty_cache()

    toc = time.time()
    sm_str = ""
    for x in stat_meters.keys():
        sm_str += f"[{x}:{stat_meters[x].avg:.1f}]" if "n_" in x else f"[{x}:{stat_meters[x].avg:.4f}]"

    logger.info(
        f"[Training]"
        f"[dataset:{args.dataset}]"
        f"[bits:{args.n_bits}]"
        f"[metric:{args.relevance_metric}]"
        f"[epoch:{epoch}/{args.n_epochs - 1}]"
        f"[time:{(toc - tic):.3f}]"
        f"{sm_str}"
    )


def train_init(args):
    # setup net
    net = build_model(args, True)

    # setup criterion
    criterion = ListwiseLoss()

    logger.info(f"number of learnable params: {calc_learnable_params(net)}")

    # setup optimizer
    optimizer = build_optimizer(args.optimizer, net.parameters(), lr=args.lr, weight_decay=args.wd)

    return net, criterion, optimizer


def train(args, train_loader, query_loader, dbase_loader):
    net, criterion, optimizer = train_init(args)

    early_stopping = EarlyStopping()

    for epoch in range(args.n_epochs):
        train_epoch(args, train_loader, net, criterion, optimizer, epoch)

        # we monitor mAP@topk validation accuracy every 5 epochs
        if (epoch + 1) % 5 == 0 or (epoch + 1) == args.n_epochs:
            early_stop = validate_smart(
                args,
                query_loader,
                dbase_loader,
                early_stopping,
                epoch,
                model=net,
                multi_thread=args.multi_thread,
            )
            if early_stop:
                break

    if early_stopping.counter == early_stopping.patience:
        logger.info(
            f"without improvement, will save & exit, best mAP: {early_stopping.best_map:.3f}, best epoch: {early_stopping.best_epoch}"
        )
    else:
        logger.info(
            f"reach epoch limit, will save & exit, best mAP: {early_stopping.best_map:.3f}, best epoch: {early_stopping.best_epoch}"
        )

    save_checkpoint(args, early_stopping.best_checkpoint)

    return early_stopping.best_epoch, early_stopping.best_map


def main():
    init()
    args = get_config()

    # =========================
    # Semantic relevance setting
    # choose one from:
    # "cosine", "binary", "jaccard", "euclidean"
    # =========================
    args.tau = 0.005
    args.relevance_metric = "binary"

    if "rename" in args and args.rename:
        rename_output(args)

    dummy_logger_id = None
    rst = []

    for dataset in ["flickr"]:
        print(f"processing dataset: {dataset}")

        args.dataset = dataset
        args.n_classes = get_class_num(dataset)
        args.topk = get_topk(dataset)

        train_loader, query_loader, dbase_loader = build_loaders(
            dataset,
            args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.n_workers
        )

        for hash_bit in [64]:
            print(
                f"processing hash-bit: {hash_bit}, "
                f"semantic relevance metric: {args.relevance_metric}"
            )

            seed_everything(args.seed)
            args.n_bits = hash_bit

            args.save_dir = f"./output/{args.backbone}/{dataset}/{hash_bit}/{args.relevance_metric}"
            os.makedirs(args.save_dir, exist_ok=True)

            if dummy_logger_id is not None:
                logger.remove(dummy_logger_id)

            dummy_logger_id = logger.add(
                f"{args.save_dir}/train.log",
                mode="w",
                level="INFO"
            )

            with open(f"{args.save_dir}/config.json", "w") as f:
                json.dump(
                    vars(args),
                    f,
                    indent=4,
                    sort_keys=True,
                    default=lambda o: o if type(o) in [bool, int, float, str] else str(type(o)),
                )

            best_epoch, best_map = train(args, train_loader, query_loader, dbase_loader)

            rst.append(
                {
                    "dataset": dataset,
                    "hash_bit": hash_bit,
                    "relevance_metric": args.relevance_metric,
                    "best_epoch": best_epoch,
                    "best_map": best_map,
                }
            )

    print_in_md(rst)


if __name__ == "__main__":
    main()
