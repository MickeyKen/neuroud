#!/usr/bin/python
import matplotlib.pyplot as plt

path = 'loss_output_for_plot.txt'
isServiceCount = False

if __name__ == '__main__':


    xp = []
    yp = []
    average_xp = []
    average_yp = []

    last_step = 0

    sum = 0
    count = 0

    fig = plt.figure()

    plt.ion()
    # plt.title('Simple Curve Graph')
    plt.xlabel('Episode')
    plt.ylabel('Serviced human count')
    plt.xlim(0,2000)
    plt.ylim(0,100)
    plt.grid()
    with open(path) as f:
        for s_line in f:
            step = int(s_line.split(',')[0])
            data = s_line.split(',')[1]
            data = data.replace('[', '')
            data = data.replace(']', '')
            data = float(data)
            # print(int(moji.split('.')[0]))
            if last_step != step:
                if last_step > 0:
                    sum = sum / float(count)
                    xp.append(step)
                    yp.append(sum)
                sum = 0
                count = 1
                sum = data

            else:
                sum += data
            count += 1

            last_step = step

        plt.plot(xp,yp, color="#a9ceec", alpha=0.5)
        plt.plot(average_xp,average_yp, color="#00529a")
        plt.draw()
        plt.pause(0)
