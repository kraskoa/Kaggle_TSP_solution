import csv


def save_best_route_to_csv(input_filepath: str, output_filepath: str):
    with open(input_filepath, "r") as file:
        lines = file.readlines()

    # Znajdź linię zawierającą najlepszą trasę
    for line in lines:
        if line.startswith("Najlepsza trasa:"):
            best_route_str = line.split(":")[1].strip()
            break

    # Przetwórz najlepszą trasę do listy identyfikatorów centroidów
    best_route = best_route_str.strip("[]").split(", ")

    # Zapisz każdy identyfikator centroidu w osobnej linii w nowym pliku CSV
    with open(output_filepath, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["ClusterId"])  # Nagłówek kolumny
        for centroid_id in best_route:
            csvwriter.writerow([centroid_id])


# Użycie funkcji
input_filepath = (
    "/home/krol/pop/pop_projekt_24z/src/best_route_10k_c07_m02_p600_3opt.txt"
)
output_filepath = "/home/krol/pop/pop_projekt_24z/src/best_route_centroids.csv"
save_best_route_to_csv(input_filepath, output_filepath)
