with open("data/the-verdict.txt", "r") as f:
    text = f.read()

print("Length of text: ", len(text))
print(text[:99])