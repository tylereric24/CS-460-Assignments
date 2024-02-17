# Do not change the filename or function headers
# You are free to add helper functions 
# note: you will only have 5 attempts to run the autograder

import numpy as np
from timeit import default_timer as timer
# import plot lib. to plot results.
# you may need to install matplotlib by typing "python3 -m pip install -U matplotlib" to test in your local machine
import matplotlib.pyplot as plt


# function header for Q1.1 (Auto & manual grading)
# input_list -> list of integers
# return -> list of integers
# note: tested input will be unordered random lists from size n=1 to n=1000
# each test input is randomized, so it is possible to pass a test once and fail the next attempt
def selection_sort(input_list):
    # Traverse through all elements in the array
    for i in range(len(input_list) - 1):
        # Assume the current index as the index of the minimum element
        min_index = i
        
        # Iterate over the unsorted part of the array to find the minimum element
        for j in range(i + 1, len(input_list)):
            # If the current element is smaller than the previously assumed minimum element
            if input_list[j] < input_list[min_index]:
                # Update the index of the minimum element
                min_index = j
        
        # If the index of the minimum element is not the same as the current index
        if min_index != i:
            # Swap the current element with the minimum element
            input_list[i], input_list[min_index] = input_list[min_index], input_list[i]
    
    return input_list

# Example usage
A = [3, 2, 10, 14]
print(selection_sort(A))  # Output: [2, 3, 10, 14]


# function header for Q1.2 (Auto & manual grading)
# input_list -> list of integers
# return -> list of integers
#
# note: tested input will be unordered random lists from size n=1 to n=1000
# each test input is randomized, so it is possible to pass a test once and fail the next attempt
def bubble_sort(input_list):
    # Traverse through all elements in the array
    for i in range(len(input_list) - 1):
        # Traverse through the unsorted part of the array in reverse order
        for j in range(len(input_list) - 1, i, -1):
            # If the current element is smaller than the previous element
            if input_list[j] < input_list[j - 1]:
                # Swap the current element with the previous element
                input_list[j], input_list[j - 1] = input_list[j - 1], input_list[j]
    
    return input_list

# Example usage
print(bubble_sort(A))  # Output: [2, 3, 10, 14]


# function header for Q1.3 (Manual grading)
# note: no inputs and outputs are needed
def q1_3():
    ''' write your answers (codes/texts) from here '''
    '''Selection Sort and Bubble Sort are both comparison-based sorting algorithms.
    They both have a time complexity of O(n^2) in the worst case. However, Selection Sort has a better performance than Bubble Sort in the average case.
    This is because Selection Sort makes fewer swaps than Bubble Sort. In the best case, both algorithms have a time complexity of O(n^2).
    However, in the best case, Bubble Sort makes fewer comparisons than Selection Sort. 
    Therefore, Bubble Sort has a better performance than Selection Sort in the best case. 
    In conclusion, Selection Sort is better than Bubble Sort in the average case, while Bubble Sort is better than Selection Sort in the best case. 
    Both algorithms have the same performance in the worst case.'''

    ''' end your answers '''
    ...


# function header for Q2.1 (Auto & manual grading)
# input_list -> list of integers
# return -> list of integers
#
# note: tested input will be unordered random lists from size n=1 to n=1000
# each test input is randomized, so it is possible to pass a test once and fail the next attempt
def insertion_sort(input_list):
    # Traverse through all elements starting from the second element
    for i in range(1, len(input_list)):
        # Store the current element to be inserted at the correct position
        current_element = input_list[i]
        
        # Find the correct position to insert the current element
        j = i - 1
        while j >= 0 and input_list[j] > current_element:
            # Shift elements greater than the current element to the right
            input_list[j + 1] = input_list[j]
            j -= 1
        
        # Insert the current element at the correct position
        input_list[j + 1] = current_element
    
    return input_list

print(insertion_sort(A))  # Output: [2, 3, 10, 14]


# function header for Q2.2 (Manual grading)
# note: no inputs and outputs are needed; Q2.2 will be manually graded
# note: this should have no errors
def q2_2():
    # generate inputs, e.g., when n=100, n=1000, n=2000, ..., n=5000
    size = [100, 1000, 2000, 3000, 4000, 5000]

    # generate different types of inputs
    random_set, asc_set, dsc_set = [], [], []
    for i in range(0, len(size)):
        random_set.append(np.random.randint(0, 1000000, size[i]))
        asc_set.append(np.arange(0, size[i]))
        dsc_set.append(np.arange(size[i], 0, -1))
    
    # measure running time for the types of inputs and save it to the following three arrays
    elapsed_time_random, elapsed_time_asc, elapsed_time_dsc = [], [], []
    ''' Write your codes to measure running time of sorting algorithms '''
    ''' Hint: you can use "start = timer()" and "end = timer()", then "end - start" to get running time '''
    ''' write your answers (codes/texts) from here '''
    for i in range(len(size)):
        start = timer()  # Start time measurement
        insertion_sort(random_set[i])  # Sort random input
        end = timer()  # End time measurement
        elapsed_time_random.append(end - start)  # Calculate elapsed time

        start = timer()
        insertion_sort(asc_set[i])  # Sort ascending input
        end = timer()
        elapsed_time_asc.append(end - start)

        start = timer()
        insertion_sort(dsc_set[i])  # Sort descending input
        end = timer()
        elapsed_time_dsc.append(end - start)



    ''' end your answers '''

    # plot the running time results
    plt.plot(size, elapsed_time_random, color='blue', label='Random input')
    plt.plot(size, elapsed_time_asc, color='red', label='asc-case')
    plt.plot(size, elapsed_time_dsc, color='green', label='dsc-case')
    plt.title("Running time of algorithms")
    plt.xlabel("Input size")
    plt.ylabel("Running time")
    plt.grid(True)
    plt.legend()
    plt.show()

q2_2()
# function header for Q2.3 (Manual grading)
# note: no inputs and outputs are needed
def q2_3():
    ''' write your answers (codes/texts) from here '''
    '''Insertion Sort has a time complexity of O(n^2) in the worst case.
    For random input the running time should grow quadratically with the input size.
    For ascending input, the running time should grow linearly with the input size.
    For descending input which is also O(n^2), the running time should grow quadratically with the input size.'''

    ''' end your answers '''
    ...


# function header for Q3.1 (Auto & manual grading)
# input_list -> list of integers
# return -> list of integers
#
# note: tested input will be unordered random lists from size n=1 to n=1000
# each test input is randomized, so it is possible to pass a test once and fail the next attempt
def merge_sort(input_list):
    # Base case: If the length of the list is 1 or less, it's already sorted
    if len(input_list) <= 1:
        return input_list
    
    # Find the middle index of the list
    mid = len(input_list) // 2
    
    # Divide the list into two halves
    left_half = input_list[:mid]
    right_half = input_list[mid:]
    
    # Recursively sort the left and right halves
    left_half = merge_sort(left_half)
    right_half = merge_sort(right_half)
    
    # Merge the sorted halves
    return merge(left_half, right_half)

def merge(left_half, right_half):
    merged_list = []
    left_index = right_index = 0
    
    # Compare elements from both halves and merge them in sorted order
    while left_index < len(left_half) and right_index < len(right_half):
        if left_half[left_index] < right_half[right_index]:
            merged_list.append(left_half[left_index])
            left_index += 1
        else:
            merged_list.append(right_half[right_index])
            right_index += 1
    
    # Append remaining elements from the left half (if any)
    while left_index < len(left_half):
        merged_list.append(left_half[left_index])
        left_index += 1
    
    # Append remaining elements from the right half (if any)
    while right_index < len(right_half):
        merged_list.append(right_half[right_index])
        right_index += 1
    
    return merged_list

# Example usage
print(merge_sort(A))  # Output: [2, 3, 10, 14]
 
# function header for Q3.2 (Manual grading)
# note: no inputs and outputs are needed; Q3.2 will be manually graded
# note: this should have no errors
def q3_2():
    # generate inputs, e.g., when n=100, n=1000, n=2000, ..., n=5000
    size = [1000, 10000, 20000, 30000, 40000, 50000]

    # generate different types of inputs
    random_set, asc_set, dsc_set = [], [], []
    for i in range(0, len(size)):
        random_set.append(np.random.randint(0, 1000000, size[i]))
        asc_set.append(np.arange(0, size[i]))
        dsc_set.append(np.arange(size[i], 0, -1))
    
    # measure running time for the types of inputs and save it to the following three arrays
    elapsed_time_random, elapsed_time_asc, elapsed_time_dsc = [], [], []
    ''' Write your codes to measure running time of sorting algorithms '''
    ''' Hint: you can use "start = timer()" and "end = timer()", then "end - start" to get running time '''
    ''' write your answers (codes/texts) from here '''
    for i in range(0, len(size)):
        start = timer()
        merge_sort(random_set[i])
        end = timer()
        elapsed_time_random.append(end - start)
        
        start = timer()
        merge_sort(asc_set[i])
        end = timer()
        elapsed_time_asc.append(end - start)
        
        start = timer()
        merge_sort(dsc_set[i])
        end = timer()
        elapsed_time_dsc.append(end - start)


    





    ''' end your answers '''
    
    # plot the running time results (no )
    plt.plot(size, elapsed_time_random, color='blue', label='Random input')
    plt.plot(size, elapsed_time_asc, color='red', label='asc-case')
    plt.plot(size, elapsed_time_dsc, color='green', label='dsc-case')
    plt.title("Running time of algorithms")
    plt.xlabel("Input size")
    plt.ylabel("Running time")
    plt.grid(True)
    plt.legend()
    plt.show()

q3_2()
# function header for Q3.3 (Manual grading)
# note: no inputs and outputs are needed
def q3_3():
    ''' write your answers (codes/texts) from here '''
    '''Merge Sort has a time complexity of O(nlogn) in all cases (best, average, and worst).
    For random input the running time should grow logarithmically with the input size. For ascending input, the running time should also grow
    logarithmically with the input size. For descending input which is also O(nlogn), the running time should also grow logarithmically with the input size.'''

    ''' end your answers '''
    ...


# function header for Q4 (Auto & manual grading)
# strs -> list of strings
# return -> list of strings
#
# note: order of the resulting list is unimportant (["ate","eat","tea"] = ["eat","ate","tea"])
#
# note: there are NO ambigous test cases where there are 2 or more resulting lists with
# an equal and largest number of anagrams (there will always only be 1 correct answer, order doesn't matter)
#
# note: to avoid ambiguity the same word is never repeated in test cases
#
# note: "First has 1, Second has 0:  'ate'" -> this notation from the autograder means that the student solution
# includes an element 'ate' but the correct solution does not
def anagram(strs):
    # Dictionary to store groups of anagrams
    anagram_groups = {}
    
    # Iterate through each string in the input list
    for s in strs:
        # Sort the characters of the string and use it as a key for grouping
        sorted_s = ''.join(merge_sort(list(s)))
        if sorted_s not in anagram_groups:
            anagram_groups[sorted_s] = []
        # Add the original string to its corresponding group
        anagram_groups[sorted_s].append(s)
    
    # Find the group with the maximum number of anagrams
    max_anagrams_group = max(anagram_groups.values(), key=len)
    
    return max_anagrams_group

# Example usage
strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
print(anagram(strs))  # Output: ["eat", "tea", "ate"]

