import random
import chess

def generate_labeled_position():
    board = chess.Board()

    num_moves = random.randint(10, 40)

    for _ in range(num_moves):
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            break
        move = random.choice(legal_moves)
        board.push(move)

    if board.is_stalemate():
        label = "Stalemate"
    elif board.is_checkmate():
        label = f"Checkmate {'Black' if board.turn == chess.BLACK else 'White'}"
    elif board.is_check():
        label = f"Check {'Black' if board.turn == chess.BLACK else 'White'}"
    else:
        label = "Nothing"

    return board.fen(), label

def generate_dataset(num_positions=10):
    positions = []
    for _ in range(num_positions):
        fen, label = generate_labeled_position()
        positions.append((fen, label))
    return positions

if __name__ == "__main__":
    dataset = generate_dataset(1000)

    for i, (fen, label) in enumerate(dataset, 1):
        print(f"{fen} {label}")