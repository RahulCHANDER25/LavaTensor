import sys
from typing import List

# Colors
GREEN = "\033[32m"
RED = "\033[31m"
BLUE = "\033[34m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
BOLD = "\033[1m"
RESET = "\033[0m"

# Emojis
CHECK = "   ✅   "
CROSS = "   ❌   "
CHESS = "♟️ "
ROBOT = "🤖"
TARGET = "🎯"
CHART = "📊"
STAR = "⭐"
TROPHY = "🏆"

# Table formatting
COL_POS = 5
COL_RES = 8
COL_AI = 17
COL_EXP = 17

def parse_fen_and_state(line: str) -> tuple:
    parts = line.strip().split()
    evaluation = parts[-1]
    is_checkmate = any(cm in line for cm in ["checkmate", "Checkmate"])
    return evaluation, is_checkmate

def read_test_positions(filename: str) -> List[tuple]:
    with open(filename, 'r') as f:
        return [parse_fen_and_state(line) for line in f if line.strip()]

def read_ai_output() -> List[str]:
    return [line.strip() for line in sys.stdin if line.strip()]

def map_expected_state(state: str, is_checkmate: bool) -> str:
    if state == "White":
        return "Checkmate White" if is_checkmate else "Check White"
    elif state == "Black":
        return "Checkmate Black" if is_checkmate else "Check Black"
    return state

def create_separator() -> str:
    """Create a separator line for the table."""
    return f"{BLUE}{'-' * COL_POS}-+-{'-' * COL_RES}-+-{'-' * COL_AI}-+-{'-' * COL_EXP}{RESET}"

def compare_states(expected_states: List[tuple], ai_states: List[str]):
    print(f"\n{BOLD}{BLUE}{CHESS} Chess Position Evaluation Report {CHESS}{RESET}")

    header = (f"{BOLD}{'#':^{COL_POS}} | {'Result':^{COL_RES}} | "
             f"{'AI Output':^{COL_AI}} | {'Expected':^{COL_EXP}}{RESET}")
    separator = create_separator()

    print(separator)
    print(header)
    print(separator)

    correct = 0
    total = len(expected_states)
    for i, ((state, is_checkmate), actual) in enumerate(zip(expected_states, ai_states), 1):
        expected = map_expected_state(state, is_checkmate)
        is_correct = actual == expected

        if is_correct:
            correct += 1
            result = f"{GREEN}{CHECK}{RESET}"
            actual_color = GREEN
            expected_color = GREEN
        else:
            result = f"{RED}{CROSS}{RESET}"
            actual_color = RED
            expected_color = YELLOW
        print(f"{CYAN}{i:^{COL_POS}}{RESET} | "
              f"{result:^{COL_RES}} | "
              f"{actual_color}{actual:^{COL_AI}}{RESET} | "
              f"{expected_color}{expected:^{COL_EXP}}{RESET}")
        if i % 10 == 0:
            print(separator)
    if total % 10 != 0:
        print(separator)
    accuracy = (correct / total) * 100 if total > 0 else 0
    accuracy_color = GREEN if accuracy >= 80 else (YELLOW if accuracy >= 50 else RED)
    print(f"\n{BOLD}{BLUE}{CHART} Summary:{RESET}")
    print(f"{STAR} Total positions: {CYAN}{total}{RESET}")
    print(f"{STAR} Correct evaluations: {GREEN}{correct}{RESET}")
    print(f"{TROPHY} Accuracy: {accuracy_color}{accuracy:.2f}%{RESET}")

def main():
    if len(sys.argv) != 2:
        print(f"{RED}Usage: python3 scripts/test.py <test_positions_file>{RESET}")
        return
    try:
        expected_states = read_test_positions(sys.argv[1])
        ai_states = read_ai_output()
        if len(expected_states) != len(ai_states):
            print(f"{RED}Error: Number of positions ({len(expected_states)}) "
                  f"does not match number of AI outputs ({len(ai_states)}){RESET}")
            return
        compare_states(expected_states, ai_states)
    except Exception as e:
        print(f"{RED}Error: {str(e)}{RESET}")

if __name__ == "__main__":
    main()