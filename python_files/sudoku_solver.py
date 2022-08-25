import numpy as np

sud = [[2, 1, 0, 0, 6, 0, 9, 0, 0], 
       [0, 0, 0, 0, 0, 9, 1, 0, 0],
       [4, 0, 9, 3, 1, 0, 0, 5, 8],
       [0, 0, 1, 0, 0, 5, 0, 4, 0],
       [9, 0, 4, 0, 3, 0, 8, 0, 5],
       [0, 5, 0, 2, 0, 0, 6, 0, 0],
       [3, 8, 0, 0, 4, 0, 5, 0, 6],
       [0, 0, 6, 7, 0, 0, 0, 0, 2],
       [0, 0, 7, 0, 8, 0, 3, 0, 9]]

def find_empty(board):
    for r in range(len(board)):
        for c in range(len(board[0])):
            if board[r][c] == 0:
                return (r,c)
    return None


def isValid(board, num, pos):
    curRow,curCol = pos
    # check row
    for c in range(len(board[0])):
        if board[curRow][c] == num and curCol != c:
            return False
    
    #checks cols
    for r in range(len(board)):
        if board[r][curCol] == num and curRow != r:
            return False
    
    posX = curCol//3 # 0,1,2
    posY = curRow//3

    for row in range(3*posY,3*posY+3):
        for col in range(3*posX, 3*posX+3):
            if board[row][col] == num and (row,col) != pos:
                return False
    return True


def solve(board):
    find = find_empty(board)
    if not find:
        return True
    else:
        row, col = find
    
    
    for i in range(1,10):
        if isValid(board, i, (row,col)):
            board[row][col] = i
            
            if solve(board):
                return True
            board[row][col] = 0
    return False

if solve(sud):
    print(sud)
else:
    print("L")

