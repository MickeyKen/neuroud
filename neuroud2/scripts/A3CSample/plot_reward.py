#!/usr/bin/python
import matplotlib.pyplot as plt

path = 'output_for_plot.txt'

if __name__ == '__main__':

    plt.ion()
    plt.title('Simple Curve Graph')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.xlim(0,600)
    plt.ylim(-100,2000)
    plt.grid()
    xp = []
    yp = []

    count = 0
    with open(path) as f:
        for s_line in f:
            moji = s_line.split(',')[2]
            # print(int(moji.split('.')[0]))
            xp.append(count + 1)
            yp.append(int(moji.split('.')[0]))
            count += 1

    plt.plot(xp,yp, color="blue")
    plt.draw()
    plt.pause(0)
