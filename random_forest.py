import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import time


def main():
    train_data = np.load(".\\data\\training_set.npz")
    train_images = train_data['images']
    train_labels = train_data['labels']

    test_data = np.load(".\\data\\test_set.npz")
    test_images = test_data['images']
    test_labels = test_data['labels']

    accuracy_list = []
    for k in range(1, 31):
        print(k)
        random_forest = RandomForestClassifier(
            n_estimators=k
        )
        random_forest.fit(
            X=train_images,
            y=train_labels
        )

        accuracy = accuracy_score(test_labels,
                                  random_forest.predict(test_images))
        accuracy_list.append(accuracy)

        print("acc: {0}".format(accuracy))

    with open("random_forest_accuracy.dat", "wb") as file:
        pickle.dump(accuracy_list, file)


if __name__ == '__main__':
    main()
