import pandas as pd


def save_new_medicine(info):

    csv_path = "dataset/Medicine_Details.csv"

    df = pd.read_csv(csv_path)

    # Prevent duplicates
    if info["medicine_name"].lower() in (
            df["Medicine Name"].str.lower().values):
        print("Medicine already exists.")
        return

    new_row = {
        "Medicine Name": info["medicine_name"],
        "Composition": info["composition"],
        "Uses": info["uses"],
        "Side_effects": info["side_effects"],
        "Manufacturer": info["manufacturer"]
    }

    df.loc[len(df)] = new_row

    df.to_csv(csv_path, index=False)

    print("New medicine added to dataset.")