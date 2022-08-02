'''
todo: add INT8 quantization implementation
'''
import os, pickle, pdb
from typing import Optional
from tqdm import tqdm
import torch
import torch.nn as nn

import torchvision.transforms as transforms

from dataloader import *
from vgg_11 import *

EPOCH = 100
LEARNING_RATE = 0.001
CHECK_POINT = 1
CLASSES = 3
NET = "vgg11"
FOLDER = "vgg11_adam_step/"


def train(model            : nn.Module,
          transform        : Optional[transforms.Compose],
          train_dataloader : DataLoader,
          optimizer        : optim.Optimizer,
          loss_fn          : nn.Module,
          lr_scheduler     : Optional[optim.lr_scheduler._LRScheduler] = None,
          use_fp16         : bool = False,
          path             : str = None
):
    model.train()
    device = torch.device('cuda:0')
    model.to(device)
    
    if path == None:
        ini_epoch = 1
        base_lr = LEARNING_RATE
        print("Creating new checkpoint ...")
    else:
        if os.path.isfile(path):
            print(f"Loading checkpoint '{path}'")
            pretrained_file = torch.load(path)
            ini_epoch = pretrained_file['epoch']
            base_lr = pretrained_file['scheduler']
            # base_lr = LEARNING_RATE  # in case you want to re-initialize lr
            # for g in optimizer.param_groups:  # in case you want to customize lr
            #     g['lr'] = LEARNING_RATE
            model.load_state_dict(pretrained_file['state_dict'])
            optimizer.load_state_dict(pretrained_file['optimizer'])

            print(f"Loaded checkpoint '{path}' (start from epoch {ini_epoch} and lr is {base_lr})")
        else:
            print(f"No checkpoint found at '{path}'")
            exit()

    per_epoch_loss, mini_batch_loss = [], []
    lr_list, test_acc = [], []
    confusion_matrices = []
    
    scaler = torch.cuda.amp.grad_scaler.GradScaler()
    ACCUMULATION_STEP = 4
    os.mkdir(FOLDER)
    for epoch in range(ini_epoch, EPOCH + 1):
        print(f"Epoch {epoch}: ")
        sum_loss = 0.
        i = 0
        # for a full dataset training iteration, each time use a single batch
        for num_batch, (image, label) in tqdm(enumerate(train_dataloader, 1)):
            image, label = image.to(device), label.to(device)

            if use_fp16:
                with torch.cuda.amp.autocast():
                    # forward
                    output = model(image)
                    loss = loss_fn(output, label)
                    mini_batch_loss.append(loss.item())
                    sum_loss += loss.item()
                    optimizer.zero_grad()

                # backward
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(image)
                loss = loss_fn(output, label)
                loss = loss / ACCUMULATION_STEP
                sum_loss += loss.item()
                loss.backward()
                if i != 0 and i % ACCUMULATION_STEP == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    i = 0
                i += 1

            # if num_batch % 40 == 0:
            #     confusion_matrix, acc = test(model, transform, device)
            #     test_acc.append(acc)
            #     confusion_matrices.append(confusion_matrix)

        lr_scheduler.step()

        per_epoch_loss.append(sum_loss / num_batch)
        print(f"Avg loss @ epoch {epoch}: {per_epoch_loss[-1]}\t"
              f"learning rate is {optimizer.param_groups[0]['lr']}")

        # if epoch % 5 == 0 and epoch != ini_epoch:
        #     confusion_matrix, acc = test(model, transform, device)
        #     test_acc.append(acc)
        #     confusion_matrices.append(confusion_matrix)
            
        if epoch % CHECK_POINT == 0 and epoch != ini_epoch:
            print("saving {}".format(epoch))
            # quantized = torch.quantization.convert(model.to('cpu').eval())
            torch.save(
                {
                    "epoch": epoch, 
                    "state_dict": model.state_dict(), 
                    "optimizer": optimizer.state_dict(), 
                    "scheduler": base_lr if lr_scheduler is None else lr_scheduler.state_dict()
                }, f"./{FOLDER}/{NET}_epoch_{epoch}_fp16.pth.tar")
                
            # dump all training param
            pickle.dump(lr_list, open(f"./{FOLDER}/lr_{NET}_{epoch}.pkl", 'wb'))
            pickle.dump(mini_batch_loss, open(f"./{FOLDER}/batch_loss_{NET}_{epoch}.pkl", 'wb'))
            pickle.dump(per_epoch_loss, open(f'./{FOLDER}/epoch_loss_{NET}_{epoch}.pkl', 'wb'))
            pickle.dump(test_acc, open(f"./{FOLDER}/test_acc_{NET}_{epoch}.pkl", 'wb'))
            pickle.dump(confusion_matrices, open(f"./{FOLDER}/confusion_matrices_{NET}_{epoch}.pkl", 'wb'))
            print("saving .pkl file complete")
    print("Training complete. Saving checkpoint...")


def test(model      : nn.Module,
         transform  : transforms.Compose,
         device     : torch.device,
         label_num  : int =3
):
    model.eval()
    testdataset = Stone(num_classes=CLASSES, prefix="../stone_enhanced_zjq_3/test", transform=transform)
    stone_testloader = DataLoader(testdataset, batch_size=1, num_workers=1, shuffle=False)
    
    print("Test the model with test dataset...")
    confusion_mat = [[0 for i in range(label_num)] for j in range(label_num)]
    true = 0
    for num_batch, (img, label) in enumerate(stone_testloader, 1):
        img = img.to(device)
        result = model(img)
        result = int(torch.argmax(result).detach().cpu())
        label = int(torch.argmax(label))
        true += int(result == label)
        confusion_mat[label][result] += 1
    print(f"Test acc is {true / num_batch}")
    return confusion_mat, true / num_batch


if __name__ == "__main__":
    if not torch.cuda.is_available():
        exit()
    BATCH_SIZE = 32

    device = torch.device('cuda:0')
    stone_dataset = Stone(num_classes=CLASSES, transform=transform)
    stone_dataloader = DataLoader(stone_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    vgg11 = VGG11(3, 3).to(device)

    for i in range(50, 1150, 50):
        print(i)
        pretrained_dict = f"./vgg11_fp32_old/vgg11_epoch_{i}.pth.tar"
        pre_model = torch.load(pretrained_dict)
        # print(pre_model.keys())
        print(pre_model['scheduler'])
        # print(pre_model['optimizer'].keys())
        # pdb.set_trace()
    
