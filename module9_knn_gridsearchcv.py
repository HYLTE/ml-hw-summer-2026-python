import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

K_RANGE = range(1, 11)


class PointStorage:
    def __init__(self, n):
        self.points = np.empty((n, 2))
        self.count = 0

    def insert(self, x, y):
        self.points[self.count] = (x, y)
        self.count += 1

    def features(self):
        return self.points[:, :1]

    def labels(self):
        return self.points[:, 1].astype(int)


def best_knn(train, test):
    search = GridSearchCV(KNeighborsClassifier(), {"n_neighbors": list(K_RANGE)})
    search.fit(train.features(), train.labels())
    k = search.best_params_["n_neighbors"]
    accuracy = search.score(test.features(), test.labels())
    return k, accuracy


def read_set(size_name, set_name):
    n = int(input(f"Enter a positive integer {size_name}: "))
    storage = PointStorage(n)
    for i in range(n):
        x = float(input(f"Enter x value of {set_name} pair {i + 1}: "))
        y = int(input(f"Enter y value of {set_name} pair {i + 1}: "))
        storage.insert(x, y)
    return storage


def main():
    train = read_set("N", "training")
    test = read_set("M", "test")

    k, accuracy = best_knn(train, test)
    print(f"The best k for kNN Classification is {k}")
    print("The corresponding test accuracy is", accuracy)


if __name__ == "__main__":
    main()
