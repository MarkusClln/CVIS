count = input("Anzahl der Wiederholungen")

print("while:")
n=0

while n<count:
    if n%2:
        print("d")
    else:
        print("D")
    n+=1

print("for")

for m in range(0, count):
    if m%2:
        print("d")
    else:
        print("D")
