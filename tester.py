import numpy as np
from layout import detectChairs
from scipy import stats

def test(id):
    chairList = detectChairs(id)
    print(f'{chairList[2]}')
    stat_result = stats.describe(chairList[2][1])
    print(stat_result)
    # 2 under 500
    print(chairList[2][1][chairList[2][1] < 500])

    print('4th chair')
    print(f'{chairList[3]}')
    stat_result = stats.describe(chairList[3][1])
    print(stat_result)
    # 2 under 500
    print(chairList[3][1][chairList[3][1] < 500])
    

if __name__ == '__main__':
    id = 0
    test(id)