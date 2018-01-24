import os
import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def main():
    # training

    train_data = np.load(".\\data\\training_set.npz")
    train_images = train_data['images']
    train_labels = train_data['labels']

    print("Training random forest...")
    random_forest = RandomForestClassifier(
        n_estimators=100
    )
    random_forest.fit(
        X=train_images,
        y=train_labels
    )

    if not os.path.exists(".\\random_forest_data"):
        os.makedirs(".\\random_forest_data")

    with open(".\\random_forest_data\\random_forest.dat", 'wb') as file:
        pickle.dump(random_forest, file)
        print("Forest saved to file.")

    # evaluating

    with open(".\\random_forest_data\\random_forest.dat", 'rb') as file:
        random_forest = pickle.load(file)

    test_data = np.load(".\\data\\test_set.npz")
    test_images = test_data['images']
    test_labels = test_data['labels']

    print(
        "acc: {0}".format(
            accuracy_score(test_labels, random_forest.predict(test_images))
        )
    )


if __name__ == '__main__':
    main()
