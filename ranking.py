import csv
from sklearn.metrics import accuracy_score
import csv
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


def rank():
    len_ = [318, 318, 318, 318, 336]
    P = []
    path_ = '/media/xxx/3AF0749EF07461D5/MFCNet/result/best'
    for i in range(5):
        print(i)
        p = [0] * len_[i]
        path = path_ + '/{}/pre/pre.csv'.format(i + 1)
        csvFile = open(path, "r")
        reader = csv.reader(csvFile)
        for i in reader:
            p[int(i[2])] = (int(i[1]),)
        P = P + p
    print(len(P))
    with open(path_ + '/pre.csv', 'w') as f:
        f_csv = csv.writer(f)
        f_csv.writerows(P)
    print(P)


rank()

t = [0] * 1608
p = [0] * 1608
p1 = []
t1 = []
t3 = []

File_path2 = '/media/xxx/3AF0749EF07461D5/MFCNet/weight/best/pre.csv'
csvFile = open(File_path2, "r")
p2 = csv.reader(csvFile)
File_path3 = '/media/xxx/3AF0749EF07461D5/MFCNet/label/true.csv'
csvFile = open(File_path3, "r")
t2 = csv.reader(csvFile)

P = []

# for i in reader:
#     p[int(i[1])]=int(i[0])
for i in p2:
    p1.append(int(i[0]))
for i in t2:
    t1.append(int(i[0]))
print(len(t1))
print(len(p1))
acc = accuracy_score(t1, p1)
pre = precision_score(t1, p1, average='macro')
re = recall_score(t1, p1, average='macro')
f1 = f1_score(t1, p1, average='macro')
print('acc', acc)
print('pre', pre)
print('re', re)
print('f1', f1)
