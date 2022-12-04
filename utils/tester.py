import numpy as np
from table_lines import slope_intercept, find_intersect, change_endpoints

def test_slope_intercept():
    lines = []
    # horizontal
    lines.append([50, 45, 100, 45])
    # vertical
    lines.append([50, 45, 50, 70])
    # horizontal
    lines.append([38, 67, 50, 12])
    # general
    lines.append([30, 52, 41, 12])
    # vertical
    lines.append([35, 45, 35, 68])
    # general
    lines.append([30, 52, 31, 12])
    # general
    lines.append([30, 52, 31, 12])
    # vertical
    lines.append([35, 45, 35, 68])
    # general
    lines.append([17, 46, 29, 14])
    lines.append([20, 35, 37, 12])
    

    # print(f'lines:\n{lines}')
    return slope_intercept(lines)

def test_intersect(slopes, yins):
    pairs = list(zip(slopes, yins))
    intersect = []
    for i in range(0, len(slopes), 2):
        intersect.append(find_intersect(pairs[i], pairs[i+1]))
    return intersect

def test_endpoint():
    lines = []
    # 1. lines that are close together, test two lines in both orders
    # with positive and similar slopes, distance btw endpoints is < 10
    lines.append([17, 46, 29, 70])
    lines.append([33, 65, 50, 90])
    lines.append([33, 65, 50, 90])
    lines.append([17, 46, 29, 70])

    # with negative slopes, distance btw endpoints is < 10
    lines.append([17, 70, 29, 46])
    lines.append([33, 50, 45, 24])
    lines.append([33, 50, 45, 24])
    lines.append([17, 70, 29, 46])

    # with diff slope signs, distance btw endpoints is < 10
    lines.append([17, 70, 29, 65])
    lines.append([33, 60, 45, 63])

    # 2. same as case1 but swap line order

    # lines with intersection point outside of both
    lines.append([17, 70, 29, 65])
    lines.append([40, 68, 70, 78])

    # lines with intersection point outside only line2
    lines.append([17, 70, 29, 65])
    lines.append([26, 73, 40, 90])

    # lines with intersection point outside only line1
    lines.append([26, 73, 40, 90])
    lines.append([17, 70, 29, 65])

    # lines with intersection point on both lines
    lines.append([17, 70, 29, 65])
    lines.append([29, 65, 40, 90])

    print(f'lines:\n{lines}')

    slopes, yins = slope_intercept(lines)
    print(f'slopes:\n{slopes}')
    print(f'yins:\n{yins}')

    pairs = list(zip(slopes, yins))
    intersect = []
    for i in range(0, len(slopes), 2):
        intersect.append(find_intersect(pairs[i], pairs[i+1]))
    
    print(f'intersects:\n{intersect}')
    w, h = 250, 100
    for i in range(0, len(lines), 2):
        result = change_endpoints(lines[i], lines[i+1], slopes[i], slopes[i+1], intersect[i//2], w, h)
        if (len(result) == 1):
            lines[i+1] = [-5, -5, -5, -5]   # set line to invalid values instead of deleting it
        else:
            lines[i], lines[i+1] = result[0], result[1]
    
    print(f'lines:\n{lines}')


def matrix_tester():
    ...

if __name__ == '__main__':
    slope, y_in = test_slope_intercept()
    # print(f'slope:\n{slope}\ny_in:\n{y_in}')

    # print(f'intersects:\n{test_intersect(slope, y_in)}')
    test_endpoint()



