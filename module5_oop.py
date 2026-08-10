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


n = int(input("Enter a positive integer N: "))

storage = NumberStorage()

for i in range(n):
    number = int(input(f"Enter number {i + 1}: "))
    storage.insert(number)

x = int(input("Enter number X: "))

print(storage.search(x))
