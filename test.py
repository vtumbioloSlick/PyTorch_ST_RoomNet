import torch

def check_cuda():
    print("Checking CUDA availability...\n")
    
    if torch.cuda.is_available():
        print("CUDA is available!")
        print(f"Number of CUDA devices: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"\n--- Device {i} ---")
            print(f"Name: {torch.cuda.get_device_name(i)}")
            print(f"Memory Allocated: {torch.cuda.memory_allocated(i)} bytes")
            print(f"Memory Cached: {torch.cuda.memory_reserved(i)} bytes")
    else:
        print("CUDA is NOT available.")
        print("Reasons might include: no GPU, driver issues, or CUDA not installed properly.")

if __name__ == "__main__":
    check_cuda()
