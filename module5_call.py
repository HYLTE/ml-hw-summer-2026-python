from module5_mod import NumberStorage

n = int(input("Enter a positive integer N: "))

storage = NumberStorage()

for i in range(n):
    number = int(input(f"Enter number {i + 1}: "))
    storage.insert(number)

x = int(input("Enter number X: "))

print(storage.search(x))
