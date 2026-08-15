import numpy as np


class PointStorage:
    def __init__(self):
        self.points = np.empty((0, 2))

    def insert(self, x, y):
        self.points = np.vstack([self.points, [x, y]])

    def knn_regression(self, x, k):
        distances = np.abs(self.points[:, 0] - x)
        nearest = np.argsort(distances)[:k]
        return np.mean(self.points[nearest, 1])


def main():
    n = int(input("Enter a positive integer N: "))
    k = int(input("Enter a positive integer k: "))

    storage = PointStorage()

    for i in range(n):
        x = float(input(f"Enter x value of point {i + 1}: "))
        y = float(input(f"Enter y value of point {i + 1}: "))
        storage.insert(x, y)

    x = float(input("Enter number X: "))

    if k <= n:
        print(f"The result of {k}-NN Regression is", storage.knn_regression(x, k))
    else:
        print("Error: k must not be greater than N")


if __name__ == "__main__":
    main()
