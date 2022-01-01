# cannot call this pyscript directly, please use the test module
import sys
import matplotlib.pyplot as plt

def plot():
    day_result_path = "./test_images/day/res.txt"
    night_result_path = "./test_images/night/res.txt"
    x = ['front', 'side', 'rear']
    day_y = []
    night_y = []

    with open(day_result_path, "r") as file:
        for line in file: 
            day_y.append(float(line.strip()))
        file.close()

    with open(night_result_path, "r") as file:
        for line in file: 
            night_y.append(float(line.strip()))
        file.close()

    x_pos = [i for i, _ in enumerate(x)]

    fig, (day_ax, night_ax) = plt.subplots(2)
    fig.set_figheight(10)
    fig.set_figwidth(10)
    
    day_ax.title.set_text('COCO model capture people output from various directions at day')
    day_ax.set(xlabel="Direction", ylabel="Average Correctness(%)")
    day_ax.set_xticks(x_pos)
    day_ax.set_xticklabels(x)
    night_ax.title.set_text('COCO model capture people output from various directions at night')
    night_ax.set(xlabel="Direction", ylabel="Average Correctness(%)")
    night_ax.set_xticks(x_pos)
    night_ax.set_xticklabels(x)

    day_ax.bar(x_pos, day_y, color='blue')
    night_ax.bar(x_pos, night_y, color='blue')

    plt.subplots_adjust(hspace=0.6)
    plt.show()