import os
import torch
import wandb
import logging
# import tiktoken
from ModelData import ModelData as MD 
from BigramLanguageModel import BigramLanguageModel as BLM
# Youtube video 1:20:01


@torch.no_grad() # not intend to call .backward
def estimate_loss(model, modelData, eval_iters):
    loss_results = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = modelData.get_batch(split, log = False)
            logits, loss = model.forward(xb, yb)
            losses[k] = loss.item()
        loss_results[split] = losses.mean()
    model.train()
    return loss_results



if __name__ == '__main__':
    # torch.set_printoptions(threshold = 10000)

    # test cuda:
    logging.warning("torch.cuda.is_available(): ", torch.cuda.is_available())
    logging.warning("torch.cuda.device_count(): ", torch.cuda.device_count())
    logging.warning("torch.cuda.current_device(): ", torch.cuda.current_device())
    logging.warning("torch.cuda.device(0): ", torch.cuda.device(0))
    logging.warning("torch.cuda.get_device_name(0):", torch.cuda.get_device_name(0))
    logging.warning("")

    # init wandb:
    wandb.init(
        # Set the project where this run will be logged
        project=os.environ["PROJECT_NAME"],
        name=os.environ["RUN_NAME"] if "RUN_NAME" in os.environ else None,
        settings=wandb.Settings(
            code_dir=os.path.dirname(os.path.realpath(__file__)
            )
        )
    )

    # hyperparameters
    data_path = 'data/dataset/tiny_shakespeare/input.txt'
    batch_size = 64
    n = 256
    train_test_ratio = 0.9
    max_iters = 500
    eval_interval = 5
    learning_rate = 3e-4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_iters = 50
    max_new_tokens = 1000
    # ------------------------------------------------------

    # set manual seed for reproducable results
    torch.manual_seed(1337)

    
    modelData = MD(data_path, batch_size, n, train_test_ratio, device)

    # xb, yb = modelData.get_batch("train",log = False)

    model = BLM(modelData).to(device)
    # model.to("cuda")

    optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)

    for iter in range(max_iters):

        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0:
            losses = estimate_loss(model, modelData, eval_interval)
            logging.warning(f"step {iter}:    train loss: {losses['train']:.4f}   |   val loss: {losses['val']:.4f}")


        xb, yb = modelData.get_batch("train", log = False)
        logits, loss = model.forward(xb.to(device), yb.to(device))
        wandb.log({"train loss": losses['train']})
        wandb.log({"validation loss": losses['val']})
        # set the gradients from the previous step to zero
        optimizer.zero_grad(set_to_none=True)

        # get the gradients for all the parameters
        loss.backward()

        # update the parameters
        optimizer.step()
    logging.warning("Final loss: ", loss.item())
    wandb.finish()
        
    # # file = open('../models/model_weights.pth', 'a')
    # # file.close()
    torch.save(model.state_dict(), 'models/model_weights.pth')


