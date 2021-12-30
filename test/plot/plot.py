import sys
import matplotlib.pyplot as plt

def usage():
    print("insufficient argument!")
    print("python plot.py 'day | night'")
    exit()

if len(sys.argv) != 2:
    usage()

if sys.argv[1] != "day" and sys.argv[1] != "night":
    usage()

result_path = "../test_images/" + sys.argv[1] + "/res.txt"
x = ['front', 'side', 'rear']
y = []

# put result to y-axis
with open(result_path, "r") as file:
    for line in file: 
        y.append(float(line.strip()))
    file.close()

x_pos = [i for i, _ in enumerate(x)]

plt.bar(x_pos, y, color='blue')
plt.xlabel("Direction on " + sys.argv[1])
plt.ylabel("Average Correctness(%)")
plt.title("COCO model capture people output from various directions")

plt.xticks(x_pos, x)

plt.show()