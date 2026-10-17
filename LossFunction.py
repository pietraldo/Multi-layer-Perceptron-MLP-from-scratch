def MSE(output, expected):
    sum = 0
    for i in range(len(output)):
        y2 = output[i]
        y = expected[i]
        sum += (y2 - y) * (y2 - y)
    return sum / 2
    