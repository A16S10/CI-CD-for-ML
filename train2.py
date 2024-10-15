import os

try:
    dataset_files = os.listdir("./Data")
    print(dataset_files)
except FileNotFoundError:
    print("Error: The directory './Data' does not exist.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")



import os

# import os
# dataset_files = os.listdir("./Data")
# dataset_files




# def train_and_evaluate(dataset_path):
#     # List all files in the specified dataset path
#     dataset_files = os.listdir(dataset_path)

#     # Print each file name one below the other
#     for dataset_file in dataset_files:
#         print(dataset_file)


# # Example usage
# dataset_path = "./Results"

# train_and_evaluate(dataset_path)

