import pandas as pd

labs=['Parassitic','Wire_Straight_Defect', 'Wire_Straight_Perfect', 'Wire_Tilted_Defect', 'Wire_Tilted_Perfect', 'Null']
csv = csv = pd.read_csv("Input_Data.csv", dtype="string")

def counter(lab, count = 0):
    for label in csv.label:
        if label == lab:
            count += 1
        else:
            continue
    return count


for lab in labs:
    perc = 100 * counter(lab) / len(csv.label)
    print(f"{lab} percentage: {perc}%")
