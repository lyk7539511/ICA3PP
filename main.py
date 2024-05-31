import torch
from data.MyData import sMRInpyDataset
from torch.utils.data import DataLoader
import torch.optim as optim
from model.sMRI import sMRINet
import wandb
import random
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

# train function (input: torch.Size([n, 166, 256, 256]), output: 3 classes)
def train(model, device, train_loader, test_loader, optimizer, epochs, scheduler):
    print(model, device, train_loader, optimizer, epochs)
    for epoch in range(1, epochs + 1):
        for batch_idx, (data, target) in enumerate(train_loader):
            model.train()
            mri = data[0].float()
            apoe = data[1]
            mri, apoe, target = mri.to(device), apoe.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(mri, apoe)
            # acc
            acc = (output.argmax(dim=1) == target).float().mean()
            wandb.log({"acc": acc})
            loss = torch.nn.functional.cross_entropy(output, target)
            wandb.log({"loss": loss.item()})
            loss.backward()
            optimizer.step()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(),
                acc))
            # eval
            if batch_idx % 10 == 0 and batch_idx > 0:
                print('evaling')
                test(model, device, test_loader)
                print('eval done')
        scheduler.step()
        
    model.eval()
    Accuracy, Precision, Recall, F1 = test(model, device, test_loader)
    wandb.log({"Accuracy": Accuracy})
    wandb.log({"Precision": Precision})
    wandb.log({"Recall": Recall})
    wandb.log({"F1": F1})

# test function (input: torch.Size([n, 166, 256, 256]), output: 3 classes)
def test(model, device, test_loader):
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            mri = data[0].float()
            apoe = data[1]
            mri, apoe, target = mri.to(device), apoe.to(device), target.to(device)
            output = model(mri, apoe)
            test_loss += torch.nn.functional.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            # precision, recall, f1
            Precision, Recall, F1, _ = precision_recall_fscore_support(target.cpu().numpy(), pred.cpu().numpy(), average='macro')

    test_loss /= len(test_loader.dataset)
    Accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        Accuracy))
    
    wandb.log({"test_loss": test_loss})
    wandb.log({"test_accuracy": 100. * correct / len(test_loader.dataset)})

    return Accuracy, Precision, Recall, F1


# main function
def main():

    seed = 3407
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False

    # WandB – Initialize a new run
    wandb.init(project="sMRI")

    # Training settings
    batch_size = 32
    epochs = 2
    lr = 1e-3
    weight_decay=1e-4
    momentum = 0.5
    no_cuda = False
    log_interval = 10

    use_cuda = not no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}    

    sMRIFilePath = './data/idaSearch_with_npy_path.csv'
    dataset = sMRInpyDataset(sMRIFilePath)
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    print('train_size:', train_size)
    print('test_size:', test_size)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, **kwargs)

    model = sMRINet().to(device)

    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

    train(model, device, train_loader, test_loader, optimizer, epochs, scheduler)
    test(model, device, test_loader)

    torch.save(model.state_dict(), "sMRINet.pt")

if __name__ == '__main__':
    main()