import torch
from sklearn.cluster import KMeans
from lib.config.para import parse_opt
opt = parse_opt()
from lib.utils.utils import cosine, list_tensor


def get_all_fea(model, loader, fea_all):
    model.eval()
    for img, _, label in loader:
        img = img.cuda()
        fea = model(img, 'get')
        with torch.no_grad():
            # todo
            label = label.cuda().data.cpu()
            fea = fea.cuda().data.cpu()
            fea = fea.squeeze()
            # not 0 and not 0.333
            index_label = torch.nonzero(label,as_tuple=False)
            for i in index_label:
                if label[i[0]][i[1]]!= 1/3:
                    fea_all[i[1]].append(label[i[0]][i[1]] * fea[i[0]])
    return fea_all


def get_all_fea_wo1_3(model, loader, fea_all):
    # without label smaller than 1/3
    model.eval()
    for img, _, label in loader:
        img = img.cuda()
        fea = model(img, 'get')
        with torch.no_grad():
            # todo
            label = label.cuda().data.cpu()
            fea = fea.cuda().data.cpu()
            fea = fea.squeeze()
            # not 0 and not 0.333
            index_label = torch.nonzero(label,as_tuple=False)
            for i in index_label:
                if label[i[0]][i[1]]!= 1/3:
                    fea_all[i[1]].append(label[i[0]][i[1]] * fea[i[0]])
    return fea_all


def get_all_fea_w1_3(model, loader, fea_all):
    # with label smaller than 1/3
    model.eval()
    for img, _, label in loader:
        img = img.cuda()
        fea = model(img, 'get')
        with torch.no_grad():
            # todo
            label = label.cuda().data.cpu()
            fea = fea.cuda().data.cpu()
            fea = fea.squeeze()
            # not 0 and not 0.333
            index_label = torch.nonzero(label,as_tuple=False)
            for i in index_label:
                fea_all[i[1]].append(label[i[0]][i[1]] * fea[i[0]])
    return fea_all


def filter(data):
    weight_center = torch.mean(data, dim=0).view(1, -1)
    _,idx = torch.sort(cosine(data,weight_center),dim=0)
    disgard_fea = int( opt.disgard_rate * data.shape[0])
    idx = idx[disgard_fea:]
    data = data[idx]
    data = data.squeeze(dim=1)
    return data


def cluster_by_cosine(data, component):
    if isinstance(data, list):
        data = list_tensor(data)
    data = filter(data)  # discard
    data = torch.div(data, torch.sum(data, dim=1).view(-1, 1))    # normalization for feature
    kmeans = KMeans(n_clusters=component, algorithm='auto').fit(data)
    centers = kmeans.cluster_centers_  # center
    centers = torch.tensor(centers)

    label = kmeans.labels_
    data_list = []
    cluster_number = []
    for i in range(component):
        data_list.append(list())
    for fea, lb in zip(data, label):
        data_list[lb].append(fea)
    for i in range(component):
        cluster_number.append(len(data_list[i]))
    return centers, cluster_number


def cluster_by_cosine_nofilter(data, component):
    if isinstance(data, list):
        data = list_tensor(data)
    data = torch.div(data, torch.sum(data, dim=1).view(-1, 1))    # normalization for feature
    kmeans = KMeans(n_clusters=component, algorithm='auto').fit(data)
    centers = kmeans.cluster_centers_  # center
    centers = torch.tensor(centers)

    label = kmeans.labels_
    data_list = []
    cluster_number = []
    for i in range(component):
        data_list.append(list())
    for fea, lb in zip(data, label):
        data_list[lb].append(fea)
    for i in range(component):
        cluster_number.append(len(data_list[i]))
    return centers, cluster_number