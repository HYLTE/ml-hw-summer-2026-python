import numpy as np
from sklearn.metrics import precision_score, recall_score


class PointStorage:
    def __init__(self, n):
        self.points = np.empty((n, 2), dtype=int)
        self.count = 0

    def insert(self, x, y):
        self.points[self.count] = (x, y)
        self.count += 1

    def precision(self):
        return precision_score(self.points[:, 0], self.points[:, 1], zero_division=0)

    def recall(self):
        return recall_score(self.points[:, 0], self.points[:, 1], zero_division=0)


def read_label(prompt):
    while True:
        value = int(input(prompt))
        if value in (0, 1):
            return value
        print("Error: value must be 0 or 1")


def main():
    n = int(input("Enter a positive integer N: "))

    storage = PointStorage(n)

    for i in range(n):
        x = read_label(f"Enter x value (ground truth) of point {i + 1}: ")
        y = read_label(f"Enter y value (predicted) of point {i + 1}: ")
        storage.insert(x, y)

    print("The Precision is", storage.precision())
    print("The Recall is", storage.recall())


if __name__ == "__main__":
    main()
