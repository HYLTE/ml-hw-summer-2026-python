n = int(input("Enter a positive integer N: "))

numbers = []

for i in range(n):
    number = int(input(f"Enter number {i + 1}: "))
    numbers.append(number)

x = int(input("Enter number X: "))

if x in numbers:
    print("The index is", numbers.index(x) + 1)
else:
    print(-1)
