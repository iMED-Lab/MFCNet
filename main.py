# -*- coding: utf-8 -*-
import logging.config
import os
import numpy as np
import numpy
from torch import optim, nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset import Lmaster_train, Lmaster_test, Lmaster_val
from LabelSmoothing import LSR
import torchvision.transforms as transforms
from models.new_model import *

logger_name = __name__
log = logging.getLogger(logger_name)

if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    batch_size = 16
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10, resample=False, expand=False, center=None),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    k = 4
    index = 5
    lr = 4e-05
    w = 0.005
    print('lr=', lr, 'weight_decay=', w)
    data_loader = DataLoader(Lmaster_train(path='/media/xxx/3AF0749EF07461D5/MFCNet',
                                           File_path='/media/xxx/3AF0749EF07461D5/MFCNet/label/{}/train/{}.csv'.format(
                                               k, index),
                                           transform=train_transforms, target_transform=None),
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=16)

    validate = DataLoader(Lmaster_val(path='/media/xxx/3AF0749EF07461D5/MFCNet',
                                      File_path='/media/xxx/3AF0749EF07461D5/MFCNet/label/{}/val/{}.csv'.format(k,
                                                                                                                index),
                                      transform=test_transforms, target_transform=None),
                          batch_size=batch_size,
                          num_workers=16)
    testdate = DataLoader(Lmaster_test(path='/media/xxx/3AF0749EF07461D5/MFCNet',
                                       File_path='/media/xxx/3AF0749EF07461D5/MFCNet/label/{}/test/{}.csv'.format(k,
                                                                                                                  index),
                                       transform=test_transforms, target_transform=None),
                          batch_size=1,
                          num_workers=16)
    print('k', k, 'index', index)

    net = Mymodel()

    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=w)
    schedulers = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, last_epoch=-1)
    net = torch.nn.DataParallel(net).cuda()
    weight = torch.from_numpy(np.array([0.5, 1])).float()
    criterion = nn.CrossEntropyLoss().cuda()
    LSR_loss = LSR().cuda()

    best = {'loss': 0.0, 'save': ''}
    log.info('train image num: {}'.format(len(data_loader.dataset)))
    log.info('test image num: {}'.format(len(validate.dataset)))
    best_acc = 0.0
    for epoch in range(200):
        schedulers.step()
        ##########################  train the model  ###############################
        runing_loss = 0.0
        net.train(mode=True)
        validate_right_count = 0
        validate_right_count1 = 0

        validate_right_count2 = 0
        for i, data in enumerate(data_loader):
            input, labels, average, lens, v, a = data

            inputs, labels, average, lens, image3, a = Variable(input.cuda()), Variable(labels.cuda()), Variable(
                average.cuda()), Variable(lens.cuda()), Variable(v.cuda()), Variable(a.cuda())
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            net.zero_grad()
            loss.backward()
            optimizer.step()
            # print statistics
            runing_loss += loss.item()
            out_vis = torch.cat(
                [(lens)[0:1, :, :, :]], 0)
            outputs = numpy.argmax(outputs.cpu().data.numpy(), axis=1)
            equal = outputs.reshape([-1, 1]) == labels.cpu().data.numpy().reshape([-1, 1])
            validate_right_count += len(equal[equal])
        net.eval()
        with torch.no_grad():
            for j, validate_data in enumerate(validate):
                input, validate_labels, name, aver, lens, v, a = validate_data
                inputs, lens, image3, a, labels = Variable(input.cuda()), Variable(lens.cuda()), Variable(
                    v.cuda()), Variable(a.cuda()), Variable(validate_labels.cuda())
                validate_outputs = net(inputs)
                validate_outputs = numpy.argmax(validate_outputs.cpu().data.numpy(), axis=1)
                equal = validate_outputs.reshape([-1, 1]) == validate_labels.cpu().data.numpy().reshape([-1, 1])
                validate_right_count1 += len(equal[equal])
                out_vis = torch.cat(
                    [(lens)[0:1, :, :, :]], 0)
        test_acc = validate_right_count1 / len(validate.dataset)
        print('[%d, %5d] train loss: %f train_accuracy: %f val_accuracy: %f' %
              (epoch + 1, i + 1, runing_loss / (i + 1), validate_right_count / len(data_loader.dataset), test_acc))
        save_path0 = '/media/xxx/3AF0749EF07461D5/MFCNet/xiaorong/k4/{}/model/{}'.format(k, index)
        isExists = os.path.exists(save_path0)
        if not isExists:
            os.makedirs(save_path0)
        save_path = os.path.join(save_path0, '{}.pth'.format(index))
        if test_acc > best_acc:
            t = []
            best_acc = test_acc
            torch.save(net, save_path)
            net.eval()
            with torch.no_grad():
                for j, validate_data in enumerate(testdate):
                    input, validate_labels, name, aver, m, u, a = validate_data
                    inputs, labels, lens, image3, a = Variable(input.cuda()), Variable(
                        validate_labels.cuda()), Variable(m.cuda()), Variable(u.cuda()), Variable(a.cuda())
                    validate_outputs = net(inputs)
                    validate_outputs = numpy.argmax(validate_outputs.cpu().data.numpy(), axis=1)
                    equal = validate_outputs.reshape([-1, 1]) == validate_labels.cpu().data.numpy().reshape([-1, 1])
                    validate_right_count2 += len(equal[equal])
                    t.append((validate_outputs[0],))
                    out_vis = torch.cat(
                        [(lens)[0:1, :, :, :]], 0)
            print("test_acc:", test_acc, len(testdate.dataset))
