from data.data import load_dataset
from models import get_model
from config import get_arg_parser
from train_utils import train_step, eval, adversarial_eval, save_ckpt, load_ckpt
import torch.optim as optim
import torch.nn as nn
import numpy as np
import json
from attack_utils import get_attack_fn
import os


if __name__ == "__main__":
    parser = get_arg_parser()
    args = parser.parse_args()
    train_loader, test_loader = load_dataset(args)
    model = get_model(args)(args, train_loader)

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.9)
    best = np.inf
    if args.resume:
        model, optimizer, best, current_epoch = load_ckpt(args, model, optimizer)
    current_epoch = 0
    if args.device != "cpu":
        model.cuda()
    if args.adversarial_eval:
        with open(args.attack_config, "r") as fp:
            attack_config = json.load(fp)
        model.eval()
        attack_fn, attack_config = get_attack_fn(attack_config)
        train_res = adversarial_eval(attack_fn, model, train_loader, epsilons=[0, attack_config.epsilon], args=args)
        val_res = adversarial_eval(attack_fn, model, test_loader, epsilons=[0, attack_config.epsilon], args=args)

        res_folder = os.path.join(args.base_dir, args.dataset, args.model, f"{attack_config.attack_name}_{attack_config.constraint}_{attack_config.epsilon}")
        if not os.path.isdir(res_folder):
            os.makedirs(res_folder)
        train_path = os.path.join(res_folder, "train_result.json")
        val_path = os.path.join(res_folder, "val_result.json")
        train_res.update({"Epoch": current_epoch})
        val_res.update({"Epoch": current_epoch})
        with open(train_path, "w") as fp:
            json.dump(train_res, fp)
        with open(val_path, "w") as fp:
            json.dump(val_res, fp)

        print(json.dumps(train_res))
        print(json.dumps(val_res))

    else:
        for epoch in range(current_epoch, args.max_epochs):
            train_step(args, model, train_loader, optimizer)
            if (epoch+1) % args.eval_freq == 0:
                train_res = eval(model, train_loader, args)
                train_res.update({"Epoch": epoch+1})
                val_res = eval(model, test_loader, args)
                val_res.update({"Epoch": epoch + 1})
                print("Train", train_res)
                print("Val", val_res)
                scheduler.step(val_res["Loss"])
                if best > train_res["Loss"]:
                    save_ckpt(args, model, optimizer, epoch+1, train_res, val_res, "best")
                    best = train_res["Loss"]
                save_ckpt(args, model, optimizer, epoch + 1, train_res, val_res, "latest")

