from vggFinetuneModel import FineTuneModel
from CIFAR_DataLoader import CifarDataManager
import numpy as np

'''
For fine tune model, the data label should be different:
    if the fine tune trimmed model if for classify label:[0]
    the fine tune model data labels would be of 2
    Thus, in this case, we need to change the label correspongdingly then go to train 
'''
def one_hot(vec, vals=10):
    n = len(vec)
    out = np.zeros((n, vals))
    out[range(n), vec] = 1
    return out

def modify_label(labels, test_classes = [0]):
    test_classes.sort()
    tmp_labels = []
    for i in labels:
        if i in test_classes:
            tmp_labels.append(test_classes.index(i))
        else:
            tmp_labels.append(len(test_classes))
    return one_hot(tmp_labels, vals=len(test_classes)+1)

data_loader = CifarDataManager()
sizes = [5,10,25]
for size in sizes:
    average_random = 0
    for _ in range(50):
        average = 0
        labels_test = np.random.randint(100,size=size)
        labels_test = labels_test.tolist()
        test_images, test_labels = data_loader.test.next_batch_without_onehot(500)
        test_labels = modify_label(test_labels, test_classes = labels_test)
        model = FineTuneModel(target_class_id=labels_test)
        model.assign_weight()
        tmp = model.test_accuracy(test_images, test_labels)
        print("WITHOUT FINE-TUNE, TEST ACCURACY: " + str(tmp))
        for i in range(50):
            train_images, train_labels = data_loader.train.next_batch_without_onehot(500)
            train_labels = modify_label(train_labels, test_classes =labels_test)
            model.train_model(train_images, train_labels)
            cur = model.test_accuracy(test_images, test_labels)
            print("DURING FINE-TUNE, TEST ACCURACY: " + str(cur))
            if (i == 49):
                average += cur
        print(str(labels_test) + "ACCURACY: " + str(average/50))
        average_random += average/50
    print("AVERAGE ACCURACY FOR" + str(size)+"-CLASS: ", average_random/50)
