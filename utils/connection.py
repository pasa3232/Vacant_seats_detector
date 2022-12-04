# import numpy as np

# # return connection between nodes starting from 'start' 
# # nodes are intersections, not endpoints (index for first line, index for sec line)

# def parse_matrix(matrix, visited, start, n):
#     print(f'visited:\n{visited}')
#     print(f'start: {start}')
#     temp = []
#     return_connections = []
#     for i in range(n):
#         # print(f'start: {start}, i: {i}, visited[i]: {visited[i]}, matrix[start][i] = {matrix[start][i]}')
#         if start == i or visited[i] == 1 or matrix[start][i] == -1:
#             continue
#         # connect.append([i, matrix[start][i]])
#         temp.append([start, i])
#         print(f'temp: {temp}')
#         visited[i] = 1
#         temp_connections = parse_matrix(matrix, visited, i, n)
#         print(f'temp_connections:\n{temp_connections}')
#         for k in range(len(temp_connections)):
#             print(f'added lists:\n{temp + temp_connections[k]}')
#             temp += temp_connections[k]
#         return_connections += temp
#         visited[i] = 0
#         del temp[-1]
#     return return_connections
        


# if __name__ == '__main__':
#     n = 3
#     # matrix of intersection info
#     # matrix = [[[] for _ in range(n)] for _ in range(n)]
#     matrix = []
#     for i in range(n):
#         a = []
#         for j in range(n):
#             a.append(int(input()))
#         matrix.append(a)
    
#     print(f'matrix:\n{matrix}')

#     visited = [0 for _ in range(n)]
#     print(f'initial visited:\n{visited}')
#     all_connections = []

#     for start in range(n):     
#         visited[start] = 1
#         temp = parse_matrix(matrix, visited, start, n)
#         print(f'main temp:\n{temp}')
#         all_connections += temp
#         visited[start] = 0
    
#     print(f'all_connections:\n{all_connections}')
        
    




    