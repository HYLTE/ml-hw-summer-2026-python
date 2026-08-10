class NumberStorage:
    def __init__(self):
        self.numbers = []

    def insert(self, number):
        self.numbers.append(number)

    def search(self, x):
        for index, number in enumerate(self.numbers, start=1):
            if number == x:
                return index
        return -1
