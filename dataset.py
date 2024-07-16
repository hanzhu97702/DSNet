import torch
import torch.utils.data as Data

from scipy.io import loadmat
import numpy as np

def prepare_dataset(args, samples_type='ratio'):
    # prepare data
    if args.dataset == 'Indian':
        data = loadmat('./data/IndianPine.mat')
        TR = data['TR']
        TE = data['TE']
        input = data['input'] #(145,145,200)
    elif args.dataset == 'Berlin':
        data = loadmat('./data/Berlin/data_HS_LR.mat')
        data_train = loadmat('./data/Berlin/TrainImage.mat')
        data_test = loadmat('./data/Berlin/TestImage.mat')
        TR = data_train['TrainImage']
        TE = data_test['TestImage']
        input = data['data_HS_LR'] #(1723,476,244)
    elif args.dataset == 'Augsburg':
        data = loadmat('./data/Augsburg/data_HS_LR.mat')
        data_train = loadmat('./data/Augsburg/TrainImage.mat')
        data_test = loadmat('./data/Augsburg/TestImage.mat')
        TR = data_train['TrainImage']
        TE = data_test['TestImage']
        input = data['data_HS_LR'] #(332,485,180)
    else:
        raise ValueError("Unknown dataset")

    label = TR + TE
    num_classes = np.max(TR)
    # train data change to the ratio of train samples
    if samples_type == 'ratio':
        training_ratio = 1 # range from 0 to 1, e.g. training_ratio=0.5 means 50% training samples.
        print('Train data change to the ratio of train samples: {}'.format(training_ratio))
        train_idx, TR = split_train_data_clssnum(TR, num_classes, training_ratio)

    # normalize data by band norm
    input_normalize = np.zeros(input.shape)
    for i in range(input.shape[2]):
        input_max = np.max(input[:,:,i])
        input_min = np.min(input[:,:,i])
        input_normalize[:,:,i] = (input[:,:,i]-input_min)/(input_max-input_min)
    # data size
    height, width, band = input.shape
    print("height={0},width={1},band={2}".format(height, width, band))
    #-------------------------------------------------------------------------------
    # obtain train and test data
    total_pos_train, total_pos_test, total_pos_true, number_train, number_test, number_true = chooose_train_and_test_point(TR, TE, label, num_classes)
    mirror_image = mirror_hsi(height, width, band, input_normalize, patch=args.patches)
    x_train_band, x_test_band = train_and_test_data(mirror_image, band, total_pos_train, total_pos_test, patch=args.patches)
    y_train, y_test = train_and_test_label(number_train, number_test, num_classes)
    #-------------------------------------------------------------------------------
    # load data
    x_train=torch.from_numpy(x_train_band.transpose(0,3,2,1)).type(torch.FloatTensor) #[695, 200, 7, 7]
    y_train=torch.from_numpy(y_train).type(torch.LongTensor) #[695]
    Label_train=Data.TensorDataset(x_train, y_train)
    label_train_loader=Data.DataLoader(Label_train,batch_size=args.batch_size,shuffle=True)

    x_test=torch.from_numpy(x_test_band.transpose(0,3,2,1)).type(torch.FloatTensor) # [9671, 200, 7, 7]
    y_test=torch.from_numpy(y_test).type(torch.LongTensor) # [9671]
    Label_test=Data.TensorDataset(x_test, y_test)
    label_test_loader=Data.DataLoader(Label_test,batch_size=args.batch_size,shuffle=True)
    return label_train_loader, label_test_loader, band, height, width, num_classes, label, total_pos_true

# split dataset by training set ratio
def split_train_data_clssnum(gt, num_classes, train_num_ratio):
    train_idx = []

    TR = np.zeros_like(gt)
    for i in range(num_classes):
        idx = np.argwhere(gt == i + 1)
        samplesCount = len(idx)
        # print("Class", i, ":", samplesCount)
        sample_num = np.ceil(train_num_ratio * samplesCount).astype('int32')
        train_idx.append(idx[: sample_num])

        for j in range(sample_num):
            TR[idx[j,0], idx[j,1]] = i + 1

    train_idx = np.concatenate(train_idx, axis=0)
    return train_idx, TR

# 定位训练和测试样本
def chooose_train_and_test_point(train_data, test_data, true_data, num_classes):
    number_train = []
    pos_train = {}
    number_test = []
    pos_test = {}
    number_true = []
    pos_true = {}
    #-------------------------for train data------------------------------------
    for i in range(num_classes):
        each_class = []
        each_class = np.argwhere(train_data==(i+1))
        number_train.append(each_class.shape[0])
        pos_train[i] = each_class

    total_pos_train = pos_train[0]
    for i in range(1, num_classes):
        total_pos_train = np.r_[total_pos_train, pos_train[i]] #(695,2)
    total_pos_train = total_pos_train.astype(int)
    #--------------------------for test data------------------------------------
    for i in range(num_classes):
        each_class = []
        each_class = np.argwhere(test_data==(i+1))
        number_test.append(each_class.shape[0])
        pos_test[i] = each_class

    total_pos_test = pos_test[0]
    for i in range(1, num_classes):
        total_pos_test = np.r_[total_pos_test, pos_test[i]] #(9671,2)
    total_pos_test = total_pos_test.astype(int)
    #--------------------------for true data------------------------------------
    for i in range(num_classes+1):
        each_class = []
        each_class = np.argwhere(true_data==i)
        number_true.append(each_class.shape[0])
        pos_true[i] = each_class

    total_pos_true = pos_true[0]
    for i in range(1, num_classes+1):
        total_pos_true = np.r_[total_pos_true, pos_true[i]]
    total_pos_true = total_pos_true.astype(int)

    return total_pos_train, total_pos_test, total_pos_true, number_train, number_test, number_true

# 边界拓展：镜像
def mirror_hsi(height, width, band, input_normalize, patch=5):
    padding=patch//2
    mirror_hsi=np.zeros((height+2*padding,width+2*padding,band),dtype=float)
    #中心区域
    mirror_hsi[padding:(padding+height),padding:(padding+width),:]=input_normalize
    #左边镜像
    for i in range(padding):
        mirror_hsi[padding:(height+padding),i,:]=input_normalize[:,padding-i-1,:]
    #右边镜像
    for i in range(padding):
        mirror_hsi[padding:(height+padding),width+padding+i,:]=input_normalize[:,width-1-i,:]
    #上边镜像
    for i in range(padding):
        mirror_hsi[i,:,:]=mirror_hsi[padding*2-i-1,:,:]
    #下边镜像
    for i in range(padding):
        mirror_hsi[height+padding+i,:,:]=mirror_hsi[height+padding-1-i,:,:]

    print("**************************************************")
    print("patch is : {}".format(patch))
    print("mirror_image shape : [{0},{1},{2}]".format(mirror_hsi.shape[0],mirror_hsi.shape[1],mirror_hsi.shape[2]))
    print("**************************************************")
    return mirror_hsi

# 获取patch的图像数据
def gain_neighborhood_pixel(mirror_image, point, i, patch=5):
    x = point[i,0]
    y = point[i,1]
    temp_image = mirror_image[x:(x+patch),y:(y+patch),:]
    return temp_image

# 汇总训练数据和测试数据
def train_and_test_data(mirror_image, band, train_point, test_point, patch=5, true_point=None):
    x_train = np.zeros((train_point.shape[0], patch, patch, band), dtype=float)
    x_test = np.zeros((test_point.shape[0], patch, patch, band), dtype=float)

    for i in range(train_point.shape[0]):
        x_train[i,:,:,:] = gain_neighborhood_pixel(mirror_image, train_point, i, patch)
    for j in range(test_point.shape[0]):
        x_test[j,:,:,:] = gain_neighborhood_pixel(mirror_image, test_point, j, patch)

    print("x_train shape = {}, type = {}".format(x_train.shape,x_train.dtype))
    print("x_test  shape = {}, type = {}".format(x_test.shape,x_test.dtype))
    # if true_point.all() != None:
    if true_point != None:
        x_true = np.zeros((true_point.shape[0], patch, patch, band), dtype=float)
        for k in range(true_point.shape[0]):
            x_true[k,:,:,:] = gain_neighborhood_pixel(mirror_image, true_point, k, patch)
        print("x_true  shape = {}, type = {}".format(x_true.shape,x_test.dtype))
        print("**************************************************")
        return x_train, x_test, x_true
    else:
        print("**************************************************")
        return x_train, x_test

# 标签y_train, y_test
def train_and_test_label(number_train, number_test, num_classes, number_true=None):
    y_train = []
    y_test = []
    for i in range(num_classes):
        for j in range(number_train[i]):
            y_train.append(i)
        for k in range(number_test[i]):
            y_test.append(i)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    print("y_train: shape = {} ,type = {}".format(y_train.shape,y_train.dtype))
    print("y_test: shape = {} ,type = {}".format(y_test.shape,y_test.dtype))

    if number_true != None:
        y_true = []
        for i in range(num_classes+1):
            for j in range(number_true[i]):
                y_true.append(i)
        y_true = np.array(y_true)
        print("y_true: shape = {} ,type = {}".format(y_true.shape,y_true.dtype))
        print("**************************************************")
        return y_train, y_test, y_true
    else:
        print("**************************************************")
        return y_train, y_test
