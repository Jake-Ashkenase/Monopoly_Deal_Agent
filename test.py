import numpy as np

# Example input
agent_board_prop = np.array([
    [0, 0, 0],  # row 0
    [0, 0, 0],  # row 1
    [2, 0, 0],  # row 2 (3rd row)
    [0, 0, 0],  # row 3
    [0, 0, 0],  # row 4
    [0, 0, 0],  # row 5
    [0, 0, 0],  # row 6
    [0, 0, 0],  # row 7
    [0, 0, 0],  # row 8
    [0, 0, 0]   # row 9
])


rent_prices = np.array([
            [0, 2, 4, 7, 7], # P_Green
            [0, 3, 8, 8, 8], # P_DBlue
            [0, 1, 2, 2, 2], # P_Brown
            [0, 1, 2, 3, 3], # P_LBlue
            [0, 1, 3, 5, 5], # P_Orange
            [0, 1, 2, 4, 4], # P_Pink
            [0, 1, 2, 3, 4], # P_Black
            [0, 2, 3, 6, 6], # P_Red
            [0, 1, 2, 2, 2], # P_Tan
            [0, 2, 4, 6, 6]  # P_Yellow
        ])

# Adjust index_array to be zero-based for proper indexing
selected_values = rent_prices[np.arange(10), agent_board_prop[:, 0]]

print(selected_values)
