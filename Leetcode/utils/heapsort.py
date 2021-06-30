# TODO ReviewMe

# 堆排序
# parent = floor((i-1)/2) left = 2i+1 right = 2i+2

def heapify(arr, n, i): 
    largest = i  
    l = 2 * i + 1     # left = 2*i + 1 
    r = 2 * i + 2     # right = 2*i + 2 
    if l < n and arr[i] < arr[l]: 
        largest = l 
    if r < n and arr[largest] < arr[r]: 
        largest = r 
    if largest != i: 
        arr[i],arr[largest] = arr[largest],arr[i]  # 交换
        heapify(arr, n, largest) 

def heapSort(arr):
    heapSize = len(arr)
    for i in range((heapSize-1)//2,-1,-1):  # 创建大顶堆
        heapify(arr,heapSize,i)
    for i in range(heapSize-1,0,-1):
        arr[i],arr[0]=arr[0],arr[i]
        heapify(arr,i,0)


if __name__ == "__main__":
    mylist = [2, 5, 3, 1, 10, 4]
    print(mylist)
    heapSort(mylist)
    print(mylist)
