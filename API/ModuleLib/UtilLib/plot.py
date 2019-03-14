# -*- coding: utf-8 -*-
#!/usr/bin/env python

import glob
import itertools
import os
import re
from textwrap import wrap

#import matplotlib
from matplotlib import figure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import tfplot
from sklearn.metrics import confusion_matrix


def split_path(path):
    '''
    'a/b/c.wav' => ('a/b', 'c', 'wav')
    :param path: filepath = 'a/b/c.wav'
    :return: basename, filename, and extension = ('a/b', 'c', 'wav')
    '''
    basepath, filename = os.path.split(path)
    filename, extension = os.path.splitext(filename)
    return basepath, filename, extension


def remove_all_files(path):
    files = glob.glob('{}/*'.format(path))
    for f in files:
        os.remove(f)


def plot_confusion_matrix(correct_labels, predict_labels, labels, labelstrings,tensor_name='confusion_matrix', normalize=False, plot=False):
    '''
    Parameters:
        correct_labels                  : These are your true classification categories.
        predict_labels                  : These are you predicted classification categories
        labels                          : This is a list of labels which will be used to display the axix labels
        title='Confusion matrix'        : Title for your matrix
        tensor_name = 'MyFigure/image'  : Name for the output summay tensor
    Returns:
        summary: TensorFlow summary
    Other itema to note:
        - Depending on the number of category and the data , you may have to modify the figzie, font sizes etc.
        - Currently, some of the ticks dont line up due to rotations.
    '''
    cm = confusion_matrix(correct_labels, predict_labels, labels=labels)
    '''
    Returns
    -------
    C : array, shape = [n_classes, n_classes]
        Confusion matrix
    Examples
    --------
    >>> from sklearn.metrics import confusion_matrix
    >>> y_true = [2, 0, 2, 2, 0, 1]
    >>> y_pred = [0, 0, 2, 2, 0, 2]
    >>> confusion_matrix(y_true, y_pred)
    array([[2, 0, 0],
           [0, 0, 1],
           [1, 0, 2]])
    '''
    if normalize:
        cm = cm.astype('float') * 10 / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm, copy=True)
        cm = cm.astype('int')

    np.set_printoptions(precision=2)
    '''
    Parameters
    ----------
    precision : int or None, optional
        Number of digits of precision for floating point output (default 8).
        May be `None` if `floatmode` is not `fixed`, to print as many digits as
        necessary to uniquely specify the value.
    保持两位小数的精度,这里也没有指定对象啊？？
    '''
    ###fig, ax = matplotlib.figure.Figure()

    if plot:
        plot_cm(cm=cm,classes=labelstrings)

    #为什么不直接用plt？因为为了通过tfplot把cm的图写进event file，后面那个to_summary()函数只能接收Figure的实例
    fig = figure.Figure(figsize=(7, 7), dpi=320, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)
    '''
    Returns
    -------
    axes : an `.axes.SubplotBase` subclass of `~.axes.Axes` (or a \
           subclass of `~.axes.Axes`)

        The axes of the subplot. The returned axes base class depends on
        the projection used. It is `~.axes.Axes` if rectilinear projection
        are used and `.projections.polar.PolarAxes` if polar projection
        are used. The returned axes is then a subplot subclass of the
        base class.
    返回一个类似pyplot的东西，拥有plt的所有属性和函数，ax.show()就相当于plt.imshow()
    '''
    ax.imshow(cm, cmap='Oranges')

    classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x) for x in labelstrings]
    classes = ['\n'.join(wrap(l, 40)) for l in classes]

    tick_marks = np.arange(len(classes))

    ax.set_xlabel('Predicted', fontsize=7)
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, fontsize=4, rotation=-90, ha='center')
    '''
    ticks是轴刻度，tick_marks=[0,1,2,3,4,5,6],class=['lable0','lable1','lable2','lable3','lable4','lable5','lable6']
    表示对应刻度tick_marks[i]处显示的值时labeli而不再是数字i
    '''
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    ax.set_ylabel('True Label', fontsize=7)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=4, va='center')
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], 'd') if cm[i, j] != 0 else '.', horizontalalignment="center", fontsize=6,
                verticalalignment='center', color="black")
    fig.set_tight_layout(True)

    #上面都是对图的设置
    summary = tfplot.figure.to_summary(fig, tag=tensor_name)
    '''
    Convert a matplotlib figure ``fig`` into a TensorFlow Summary object
    that can be directly fed into ``Summary.FileWriter``.
    Example:

      >>> fig, ax = ...    # (as above)
      >>> summary = to_summary(fig, tag='MyFigure/image')

      >>> type(summary)
      tensorflow.core.framework.summary_pb2.Summary
      >>> summary_writer.add_summary(summary, global_step=global_step)
    实际上是把plot的对象当成一个图写进summary了
    '''
    return summary


def test():
    """
    test cases for plotting confusion_matrix
    :return:
    """
    correct_labels = [1,2,2,2,0,3]
    predict_labels = [2,2,3,1,0,1]
    labels = ['AppLe','huaWei','meiZU','xi AOmi']
    classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x) for x in labels]
    print(classes)
    classes = ['\n'.join(wrap(l, 40)) for l in classes]
    print(classes)
    cm = plot_confusion_matrix(correct_labels=correct_labels,predict_labels=predict_labels,labels=None,labelstrings=classes,plot=True)
    print(cm)


def plot_cm(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    #下面是为了在每个方格中央显示具体数字，如：1表示预测label是x 真实label是y的实例的个数为1
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    '''
    Automatically adjust subplot parameters to give specified padding
    '''
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    # plt.savefig('confusion_matrix',dpi=200)


if __name__=='__main__':
    test()