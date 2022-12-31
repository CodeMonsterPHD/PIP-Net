
import argparse
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib
import torch
import numpy as np

def args_parser():
    parser = argparse.ArgumentParser(description='tsne instruction and visualization')
    parser.add_argument('-perplexity', default=10)
    parser.add_argument('--data',default=torch.randn(500,10))
    parser.add_argument('--label_state',default='no')
    parser.add_argument('--class_number',default=4)
    parser.add_argument('--label',default=np.random.randint(0,4,[500,1]))
    args = parser.parse_args()
    return args


def tsne(all_fea):
    """
    :param all_fea: data   nxm
    :return: a array which is samples x 2 or 3
    """

    # n_components is descending into 2 dimensions
    # perplexity is a guessing: maybe one class has 10 samples

    tsne = TSNE(n_components=2, perplexity=5, random_state=0)
    X_d = tsne.fit_transform(all_fea)
    # note: you get X_2d is samples x 2 array.
    return X_d


def draw(x,label_state,label,classification):
    """
    :param x: you data which need to visualization
    :param label_state: yes or no,if you have labels, you should input yes
    :param label: your labels    samples x 1
    :param classification: how many class do you have   a number
    :return:none
    """
    if label_state == 'no':
        plt.scatter(x[:,0],x[:,1],c='r')
    # we have label like:  a->14   means  a is 14 class
    if label_state == 'yes':
        color = ['black','red','purple','green','blue','pink','olive','orange','brown','steelblue','indigo','navy','cyan','slategray',
                 'blueviolet','tomato','orchid','springgreen','tan','gold','rosybrown','royalblue','darkred','khaki','palegreen','deeppink','thistle','salmon']
        # so we need to get some color to description the different dot
        # for name, hex in matplotlib.colors.cnames.items():
        #     if len(color) < classification:
        #         color.append(name)
        # draw
        for i,pairs in enumerate(x):
            label_ = label[i]
            # c is color of label of i-th sample
            plt.scatter(pairs[0],pairs[1],c=color[label_])
    f = plt.gcf()
    plt.show()


def start(args):
    # here is your data, tensor/array/list ...

    # tsne
    fea = tsne(args.data)

    # draw
    draw(fea,args.label_state,args.label,args.class_number)


if __name__ == '__main__':
    cen_label = []
    for i in range(28):
        cen_label.append(i)
        cen_label.append(i)
        cen_label.append(i)
    cen = torch.load("/disk/yk/Queue_Intent/network/pro.pth")
    # model_out为CUDA上的tensor
    model_out = cen.cpu()
    # detach()：去除梯度
    model_out = model_out.clone().detach()

    ff = tsne(model_out)
    draw(ff, 'yes', cen_label, 28)