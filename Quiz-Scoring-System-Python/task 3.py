import pandas as pd

data = []

n = int(input("How many persons? "))

for i in range(n):
    print(f"\nPerson {i + 1}")

    name = input("Enter Name: ")
    
    while True:
        try:
            age = int(input("Enter Age: "))
            break
        except ValueError:
            print("Please enter a valid number.")

    email = input("Enter Email: ")

    data.append({
        "Name": name,
        "Age": age,
        "Email": email
    })

df = pd.DataFrame(data)

file_name = "user_details.xlsx"
df.to_excel(file_name, index=False)

print(f"\nDetails saved successfully in '{file_name}'!")