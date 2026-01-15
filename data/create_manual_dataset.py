import json
import os

# Daftar semua kartu bridge yang valid
VALID_CARDS = {
    r + s for r in ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
    for s in ['S', 'H', 'D', 'C']
}

def validate_hand(hand):
    """Validasi bahwa tangan memiliki 13 kartu unik dan valid."""
    if len(hand) != 13:
        return False, "Harus ada tepat 13 kartu"
    for card in hand:
        if card not in VALID_CARDS:
            return False, f"Kartu tidak valid: {card}"
    if len(set(hand)) != 13:
        return False, "Ada kartu duplikat dalam satu tangan"
    return True, ""

def input_hand(prompt):
    """Input dan validasi tangan dari pengguna"""
    while True:
        print(prompt)
        print("Masukkan 13 kartu (format: AS KH QD ...) dipisahkan spasi:")
        cards = input().strip().split()
        if len(cards) != 13:
            print(f"Jumlah kartu salah. Harus 13 kartu (Anda masukkan: {len(cards)})")
            continue
        valid, msg = validate_hand(cards)
        if not valid:
            print("Error:", msg)
            continue
        return cards

def input_contract():
    """Input kontrak seperti '3NT' atau '4H'"""
    while True:
        contract = input("Masukkan kontrak (contoh: 3NT, 4H): ").strip()
        if len(contract) < 2:
            print("Kontrak tidak valid.")
            continue
        level = contract[0]
        suit = contract[1:]
        if level not in "1234567":
            print("Level kontrak harus antara 1–7.")
            continue
        if suit not in ["C", "D", "H", "S", "NT"]:
            print("Suit tidak valid. Harus C, D, H, S, atau NT.")
            continue
        return contract

def main():
    dataset_path = "data/raw/bridge_dataset.json"

    # Muat dataset lama jika sudah ada
    if os.path.exists(dataset_path):
        with open(dataset_path, "r") as f:
            try:
                dataset = json.load(f)
            except json.JSONDecodeError:
                dataset = []
    else:
        dataset = []

    board_id = len(dataset) + 1

    print(f"\n--- Input Board ID {board_id} ---")

    while True:
        hand1 = input_hand("\nHand 1 (North):")
        hand2 = input_hand("\nHand 2 (South):")

        # Cek duplikasi antar hand
        common = set(hand1) & set(hand2)
        if common:
            print(f"Error: Ada kartu duplikat antara hand1 dan hand2: {common}")
            print("Silakan ulangi input.")
        else:
            break

    contract = input_contract()

    entry = {
        "board_id": board_id,
        "hand1": hand1,
        "hand2": hand2,
        "contract": contract
    }

    dataset.append(entry)

    # Simpan ke file
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    with open(dataset_path, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"\nData berhasil disimpan ke {dataset_path}")
    print(f"Board {board_id} ditambahkan.")

if __name__ == "__main__":
    main()