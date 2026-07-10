a = 25
b = 40

print("--- Before Swapping ---")
print(f"Variable 1: {a}")
print(f"Variable 2: {b}\n")

a = a + b  
b = a - b 
a = a - b 

print("--- After Swapping ---")
print(f"Variable 1: {a}")
print(f"Variable 2: {b}")