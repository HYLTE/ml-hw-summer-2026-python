import numpy as np
from sklearn.neighbors import KNeighborsRegressor


class PointStorage:
    def __init__(self, n):
        self.points = np.empty((n, 2))
        self.count = 0

    def insert(self, x, y):
        self.points[self.count] = (x, y)
        self.count += 1

    def knn_regression(self, x, k):
        model = KNeighborsRegressor(n_neighbors=k)
        model.fit(self.points[:, :1], self.points[:, 1])
        return model.predict([[x]])[0]

    def labels_variance(self):
        return self.points[:, 1].var()


def main():
    n = int(input("Enter a positive integer N: "))
    k = int(input("Enter a positive integer k: "))

    storage = PointStorage(n)

    for i in range(n):
        x = float(input(f"Enter x value of point {i + 1}: "))
        y = float(input(f"Enter y value of point {i + 1}: "))
        storage.insert(x, y)

    x = float(input("Enter number X: "))

    if k <= n:
        print(f"The result of {k}-NN Regression is", storage.knn_regression(x, k))
        print("The variance of labels is", storage.labels_variance())
    else:
        print("Error: k must not be greater than N")


if __name__ == "__main__":
    main()
