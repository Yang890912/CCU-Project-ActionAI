# cannot call this pyscript directly, please use the test module
import matplotlib.pyplot as plt

def plot():
    with open("predict_result.txt", "r") as res:
        counter = 1
        frame = []
        work_prob = []
        rest_prob = []
        for line in res:
            tmp = line.strip("\n").split(",")
            work_prob.append(tmp[0]) # probability of work
            rest_prob.append(tmp[1]) # probability of rest
            frame.append(counter * 15) # analysis three images by three images and also get one image every 5 frames => 3 x 5 = 15
            counter += 1
        res.close()

    plt.xlabel("Frame to get three images")
    plt.ylabel("probability of work or rest")
    plt.title("Prediction model for state of work or rest")
    plt.scatter(frame, work_prob)
    plt.scatter(frame, rest_prob)
    plt.plot(frame, work_prob, '-o', label = "work prob")
    plt.plot(frame, rest_prob, '-o', label = "rest prob")
    plt.gca().invert_yaxis()
    plt.legend()
    plt.show()