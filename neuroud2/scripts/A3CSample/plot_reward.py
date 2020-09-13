#!/usr/bin/python
import matplotlib.pyplot as plt

path = 'output_for_plot.txt'
isServiceCount = False

if __name__ == '__main__':


    xp = []
    yp = []
    average_xp = []
    average_yp = []

    ave_num = 10

    flag = 0
    sum = 0

    count = 0

    if isServiceCount:
        plt.ion()
        plt.title('Simple Curve Graph')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.xlim(0,500)
        plt.ylim(-0,4)
        plt.grid()
        with open(path) as f:
            for s_line in f:
                moji = s_line.split(',')[4]
                # print(int(moji.split('.')[0]))
                xp.append(count + 1)
                yp.append(int(moji))

                num10 = count // ave_num

                if num10 == flag:
                    sum += int(moji)
                else:
                    sum = float(sum / float(ave_num))
                    average_xp.append((flag+1)*ave_num + 1)
                    average_yp.append(sum)
                    sum = 0
                    sum += int(moji.split('.')[0])
                    flag += 1
                count += 1

            plt.plot(xp,yp, color="#a9ceec", alpha=0.5)
            plt.plot(average_xp,average_yp, color="#00529a")
            plt.draw()
            plt.pause(0)

    else:
        plt.ion()
        plt.title('Simple Curve Graph')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.xlim(0,500)
        plt.ylim(-3000,1000)
        plt.grid()
        with open(path) as f:
            for s_line in f:
                moji = s_line.split(',')[2]
                # print(int(moji.split('.')[0]))
                xp.append(count + 1)
                yp.append(int(moji.split('.')[0]))

                num10 = count // ave_num

                if num10 == flag:
                    sum += int(moji.split('.')[0])
                else:
                    sum = sum / ave_num
                    average_xp.append((flag+1)*ave_num + 1)
                    average_yp.append(sum)
                    sum = 0
                    sum += int(moji.split('.')[0])
                    flag += 1
                count += 1

            plt.plot(xp,yp, color="#a9ceec", alpha=0.5)
            plt.plot(average_xp,average_yp, color="#00529a")
            plt.draw()
            plt.pause(0)
