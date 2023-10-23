# -*- coding: utf-8 -*-
import numpy
from torch.autograd import Variable
from dataset import *
import csv
import torchvision.transforms as transforms


# from until import *
def te(net, validate, index):
    net.eval()
    x = []
    y = []
    z = []
    num1 = []
    num2 = []
    num3 = []
    num4 = []
    o = []
    i = []
    oi = []
    io = []
    index4 = []
    save_path_ = '/media/xxx/3AF0749EF07461D5/MFCNet'
    with torch.no_grad():
        for j, validate_data in enumerate(validate):
            validate_input, validate_labels, name, l, num, imgs = validate_data
            num = num.cpu().data.numpy()[0]
            ture = validate_labels.cpu().data.numpy()
            validate_inputs = Variable(validate_input.cuda())
            validate_outputs = net(validate_inputs)
            validate_outputs = numpy.argmax(validate_outputs.cpu().data.numpy(), axis=1)

            equal = validate_outputs.reshape([-1, 1]) == validate_labels.cpu().data.numpy().reshape([-1, 1])
            t0 = (validate_outputs[0],)
            t0 = name + t0
            x.append(t0)
            if validate_outputs[0] == 0:
                t1 = (index,)
                n1 = (num,)
                t1 = t1 + n1
                o.append(t1)
                num1.append(len(x))
                if ture == 1:
                    t2 = (validate_outputs[0],)
                    t2 = name + t2
                    oi.append(t2)
                    num2.append(len(x))
            if validate_outputs[0] == 1:
                n3 = (num,)
                t3 = (validate_outputs[0],)
                if index == 4:
                    t3 = (index + 1,)
                t3 = name + t3 + n3
                i.append(t3)
                num3.append(len(x))
                if ture == 0:
                    t4 = (validate_outputs[0],)
                    t4 = name + t4
                    io.append(t4)
                    num4.append(len(x))
            with open('/media/xxx/3AF0749EF07461D5/as-oct/five_k/1-6/1/pre/pre{}.csv'.format(index), 'w') as f:
                f_csv = csv.writer(f)
                f_csv.writerows(i)
            with open('/media/xxx/3AF0749EF07461D5/as-oct/five_k/1-6/1/pre/{}.csv'.format(index), 'w') as f:
                f_csv = csv.writer(f)
                f_csv.writerows(o)

            if validate_outputs[0] != validate_labels[0]:
                y.append([name[0], validate_outputs[0], validate_labels[0]])
                if validate_outputs[0] == '1' and int(validate_labels[0]) == '2':
                    z.append([name[0], validate_outputs[0], validate_labels[0]])
    return y


def te1(net, validate, index, save_path):
    net.eval()
    x = []
    y = []
    z = []
    num1 = []
    num2 = []
    num3 = []
    num4 = []
    o = []
    i = []
    oi = []
    io = []

    with torch.no_grad():
        for j, validate_data in enumerate(validate):
            validate_input, validate_labels, name, num = validate_data
            num = num.cpu().data.numpy()[0]
            ture = validate_labels.cpu().data.numpy()
            validate_inputs = Variable(validate_input.cuda())
            validate_outputs = net(validate_inputs)
            validate_outputs = numpy.argmax(validate_outputs.cpu().data.numpy(), axis=1)

            t0 = (validate_outputs[0],)
            t0 = name + t0
            x.append(t0)

            if validate_outputs[0] == 0:
                t1 = (index,)
                if index == 4:
                    t1 = (1,)
                if index == 3:
                    t1 = (4,)
                n1 = (num,)
                t1 = name + t1 + n1
                o.append(t1)
                num1.append(len(x))
                if ture == 1:
                    t2 = (validate_outputs[0],)
                    t2 = name + t2
                    oi.append(t2)
                    num2.append(len(x))
            if validate_outputs[0] == 1:
                n3 = (num,)
                t3 = (index,)
                if index == 2:
                    t3 = (index + 1,)
                elif index == 3:
                    t3 = (index + 2,)
                elif index == 4:
                    t3 = (index - 2,)
                t3 = name + t3 + n3
                i.append(t3)
                num3.append(len(x))
                if ture == 0:
                    t4 = (validate_outputs[0],)
                    t4 = name + t4
                    io.append(t4)
                    num4.append(len(x))
            with open(save_path + '/pre{}.csv'.format(index), 'w') as f:
                f_csv = csv.writer(f)
                f_csv.writerows(i)
            with open(save_path + '/{}.csv'.format(index), 'w') as f:
                f_csv = csv.writer(f)
                f_csv.writerows(o)
        pre_path = save_path + '/pre{}.csv'.format(index)
        p_path = save_path + '/{}.csv'.format(index)
    return pre_path, p_path, i, o


if __name__ == '__main__':

    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    k = 5
    print('begin test {}'.format(k))
    pre = []
    p = []
    result_0 = []
    result_1 = []
    result_2 = []
    result_3 = []
    result_4 = []
    result_5 = []
    save_path = '/media/xxx/3AF0749EF07461D5/MFCNet/result/best/{}/pre'.format(k)
    isExists = os.path.exists(save_path)
    if not isExists:
        os.makedirs(save_path)
    for i in range(1, 6):
        model_path = '/media/xxx/3AF0749EF07461D5/MFCNet/weight/best/{}/model/{}/{}.pth'.format(k, i, i)
        net = torch.load(model_path)
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        net.cuda().eval()
        if i == 1:
            validate = DataLoader(dataset=as_oct1(root='/media/xxx/3AF0749EF07461D5/MFCNet/Lmaster_images',
                                                  File_path='/media/xxx/3AF0749EF07461D5/FLX/label/{}/k{}_test.csv'.format(
                                                      k, k),
                                                  transform=test_transforms),
                                  batch_size=1,
                                  shuffle=False,
                                  num_workers=16
                                  )
            pre_path, p_path, i, o = te1(net, validate, 0, save_path)
            pre.append(pre_path)
            p.append(p_path)
            result_0.append(o)
        if i == 2:
            validate = DataLoader(dataset=as_oct1(root='/media/xxx/3AF0749EF07461D5/MFCNet/Lmaster_images',
                                                  File_path=pre[0],
                                                  transform=test_transforms),
                                  batch_size=1,
                                  shuffle=False,
                                  num_workers=16
                                  )
            pre_path, p_path, i, o = te1(net, validate, 1, save_path)
            pre.append(pre_path)
            p.append(p_path)
        if i == 3:
            validate = DataLoader(dataset=as_oct1(root='/media/xxx/3AF0749EF07461D5/MFCNet/Lmaster_images',
                                                  File_path=p[1],
                                                  transform=test_transforms),
                                  batch_size=1,
                                  shuffle=False,
                                  num_workers=16
                                  )
            pre_path, p_path, i, o = te1(net, validate, 2, save_path)
            pre.append(pre_path)
            p.append(p_path)
            result_3.append(i)
        if i == 4:
            validate = DataLoader(dataset=as_oct1(root='/media/xxx/3AF0749EF07461D5/MFCNet/Lmaster_images',
                                                  File_path=pre[1],
                                                  transform=test_transforms),
                                  batch_size=1,
                                  shuffle=False,
                                  num_workers=16
                                  )
            pre_path, p_path, i, o = te1(net, validate, 3, save_path)
            pre.append(pre_path)
            p.append(p_path)
            result_4.append(o)
            result_5.append(i)
        if i == 5:
            validate = DataLoader(dataset=as_oct1(root='/media/xxx/3AF0749EF07461D5/MFCNet/Lmaster_images',
                                                  File_path=p[2],
                                                  transform=test_transforms),
                                  batch_size=1,
                                  shuffle=False,
                                  num_workers=16
                                  )
            pre_path, p_path, i, o = te1(net, validate, 4, save_path)
            pre.append(pre_path)
            p.append(p_path)
            result_1.append(o)
            result_2.append(i)

    result = result_0[0] + result_1[0] + result_2[0] + result_3[0] + result_4[0] + result_5[0]
    with open(save_path + '/pre.csv', 'w') as f:
        f_csv = csv.writer(f)
        f_csv.writerows(result)
    print('finish')
