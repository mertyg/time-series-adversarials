from data.data import load_dataset
from models import get_model
from config import get_arg_parser
from train_utils.training import train_step, eval, adversarial_eval
import torch.optim as optim

if __name__ == "__main__":
    parser = get_arg_parser()
    args = parser.parse_args()
    train_loader, test_loader = load_dataset(args)
    model = get_model(args)(args, train_loader)

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    for i in range(args.max_epochs):
        train_step(args, model, train_loader, optimizer)
        if (i+1) % 5 == 0:
            train_res = eval(model, train_loader, args)
            train_res.update({"Epoch": i+1})
            val_res = eval(model, test_loader, args)
            val_res.update({"Epoch": i + 1})
            print("Train", train_res)
            print("Val", val_res)
            scheduler.step(val_res["Loss"])

    train_res = adversarial_eval(model, train_loader)
    val_res = adversarial_eval(model, test_loader)

    for eps in train_res.keys():
        print(f"Training Robust Accuracy for eps = {eps} is {train_res[eps].avg}")
        print(f"Validation Robust Accuracy for eps = {eps} is {val_res[eps].avg}")

