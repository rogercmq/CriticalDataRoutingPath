from vggTrimmedModel import TrimmedModel
from CIFAR_DataLoader import CifarDataManager
import numpy as np
d = CifarDataManager()
sizes = [5,10,25]
for size in sizes:
    average_random = 0
    for _ in range(50):
        average = 0
        target_class_id = np.random.randint(100,size=size)
        model = TrimmedModel(target_class_id=target_class_id, multiPruning=True)
        for _ in range(50):
            test_images, test_labels = d.test.next_batch(2000)
            model.assign_weight()
            accuracy = model.test_accuracy(test_images, test_labels)
            average += accuracy
        print(str(target_class_id)+": "+ str(average/50))
        average_random += average/50
    print("AVERAGE ACCURACY FOR" + str(size)+"-CLASS: ", average_random/20)