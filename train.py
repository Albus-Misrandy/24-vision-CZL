import time
from defParameter import *
from load_dataset import *
from VGG16Network import *

start_time = time.time()

print(net)

for i in range (epoch):
    #print("-----开始第{}/{}轮训练，本轮学习率为:{}------".format((i+1),epoch,lr_scheduler.get_last_lr()))
    count_train = 0

    net.train()
    for (features,targets) in train_dataloader:
        if features.size(1) == 1:
            features = transfer_channel(features)
        features = features.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        output = net(features)

        Loss = loss(output,targets)
        Loss.backward()

        optimizer.step()

        count_train += 1
        
        if count_train % 100 == 0:
            end_time = time.time()
            print(f"训练批次{count_train}/{len(train_dataloader)},loss:{Loss.item():.3f},用时:{(end_time-start_time):.2f}")

            
    net.eval()
    with torch.no_grad():
        train_accuracy,train_loss = computer_accuracy_and_loss(net,train_dataset,train_dataloader,device=device)
        if train_accuracy >max_train_acc:
            max_train_acc = train_accuracy

        test_accuracy,test_loss = computer_accuracy_and_loss(net,test_dataset,test_dataloader,device=device)
        if test_accuracy > max_test_acc:
            max_test_acc = test_accuracy

        train_loss_lst.append(train_loss)
        train_acc_lst.append(train_accuracy)
        test_loss_lst.append(test_loss)
        test_acc_lst.append(test_accuracy)
        print(f'Train Loss.:{train_loss:.2f}' f' | Validation Loss.:{test_loss:.2f}')
        print(f'Train Acc.: {train_accuracy:.2f}%' f' | Validation Acc.: {test_accuracy:.2f}%')

    elapsed = (time.time()-start_time) / 60
    print(f'本轮训练累积用时:{elapsed:.2f} min')

    scheduler.step()

print("DONE!")