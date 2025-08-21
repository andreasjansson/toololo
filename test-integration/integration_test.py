import logging
import pytest
import asyncio
import openai
import os
import toololo

# Set up logging for tests
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger(__name__)


async def curl(args: list[str]) -> str:
    """Run curl command asynchronously.

    Args:
        args: List of arguments to pass to curl

    Returns:
        The output of the curl command
    """
    if "-m" not in args and "--max-time" not in args:
        args = args + ["--max-time", "30"]

    # Create subprocess
    process = await asyncio.create_subprocess_exec(
        "curl",
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    # Wait for the subprocess to finish and get stdout/stderr
    stdout, stderr = await process.communicate()

    if process.returncode != 0:
        return f"Error (code {process.returncode}): {stderr.decode()}"

    return stdout.decode()


@pytest.mark.asyncio
async def test_curl_speed_test():
    if not os.environ.get("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY not set")
        
    client = openai.AsyncOpenAI(
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "https://github.com/andreasjansson/toololo",
            "X-Title": "toololo"
        }
    )

    prompt = "Do a basic network speed test and analyze the results."

    async for output in toololo.Run(
        client,
        prompt,
        model="openai/gpt-5",
        tools=[curl],
    ):
        print(output)


class TowersOfHanoi:
    def __init__(self, num_disks=3):
        self.towers = [list(range(num_disks, 0, -1)), [], []]
        self.num_disks = num_disks

    def get_state(self) -> list[list[int]]:
        return self.towers

    def move(self, from_index: int, to_index: int) -> None:
        # Check for valid tower indices
        if not (0 <= from_index <= 2 and 0 <= to_index <= 2):
            raise ValueError("Tower index must be 0, 1, or 2")

        # Check if source tower is empty
        if not self.towers[from_index]:
            raise ValueError(f"Cannot move from empty tower {from_index}")

        # Check if move is valid (smaller disk on top of larger disk)
        if (
            self.towers[to_index]
            and self.towers[from_index][-1] > self.towers[to_index][-1]
        ):
            raise ValueError("Cannot place larger disk on top of smaller disk")

        # Move the disk
        disk = self.towers[from_index].pop()
        self.towers[to_index].append(disk)

    def is_complete(self) -> bool:
        return len(self.towers[2]) == self.num_disks


@pytest.mark.asyncio
async def test_single_agent_towers_of_hanoi():
    if not os.environ.get("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY not set")
        
    client = openai.AsyncOpenAI(
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )
    towers = TowersOfHanoi()

    assert not towers.is_complete()

    async for output in toololo.Run(
        client,
        messages=[
            {
                "role": "user",
                "content": "Solve this Towers of Hanoi puzzle. The goal is to move all disks from the first tower (index 0) to the third tower (index 2). You can only move one disk at a time, and you cannot place a larger disk on top of a smaller disk.",
            }
        ],
        model="openai/gpt-5",
        tools=[towers.get_state, towers.move, towers.is_complete],
    ):
        print(output)

    assert towers.is_complete()


class TicTacToe:
    def __init__(self):
        self.board: list[list[str | None]] = [
            [None for _ in range(3)] for _ in range(3)
        ]
        self.current_player = "X"
        self.winner = None
        self.game_over = False

    def get_board(self) -> list[list[str | None]]:
        return self.board

    def is_game_over(self) -> bool:
        return self.game_over

    def get_winner(self) -> str | None:
        return self.winner

    def make_move(self, row: int, col: int) -> bool:
        # Validate move
        if (
            self.game_over
            or not (0 <= row < 3 and 0 <= col < 3)
            or self.board[row][col] is not None
        ):
            return False

        # Make the move
        self.board[row][col] = self.current_player

        # Check for win
        if self.check_win():
            self.winner = self.current_player
            self.game_over = True
        # Check for draw
        elif all(self.board[r][c] is not None for r in range(3) for c in range(3)):
            self.game_over = True

        # Switch player
        self.current_player = "O" if self.current_player == "X" else "X"
        return True

    def check_win(self) -> bool:
        p = self.current_player
        b = self.board

        # Check rows, columns and diagonals
        for i in range(3):
            if (
                b[i][0] == b[i][1] == b[i][2] == p  # rows
                or b[0][i] == b[1][i] == b[2][i] == p  # columns
            ):
                return True

        return (
            b[0][0] == b[1][1] == b[2][2] == p  # diagonal
            or b[0][2] == b[1][1] == b[2][0] == p  # diagonal
        )

    def print_board(self) -> None:
        for i, row in enumerate(self.board):
            print(" | ".join([cell if cell else " " for cell in row]))
            if i < len(self.board) - 1:
                print("-" * 9)


@pytest.mark.asyncio
async def test_multiagent_tictactoe():
    if not os.environ.get("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY not set")
        
    client = openai.AsyncOpenAI(
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )
    game = TicTacToe()

    x_prompt = "You are player X"
    o_prompt = "You are player O"
    system_prompt = "You're playing a game of Tic-Tac-Toe. The other player will automatically make moves in between your moves. Keep playing until there's a winner or a draw"

    print("=== Starting Tic-Tac-Toe Game ===")
    game.print_board()

    tools = [
        game.get_board,
        game.make_move,
        game.is_game_over,
        game.get_winner,
    ]

    def create_generator(prompt):
        return toololo.Run(
            client,
            messages=prompt,
            model="openai/gpt-5",
            tools=tools,
            system_prompt=system_prompt,
        )

    x_generator = create_generator(x_prompt)
    o_generator = create_generator(o_prompt)

    while not game.is_game_over():
        current_player = game.current_player
        current_gen = x_generator if current_player == "X" else o_generator

        try:
            output = await anext(current_gen)
        except StopAsyncIteration:
            # Reinitialize the stopped generator
            if current_player == "X":
                x_generator = create_generator(x_prompt)
                current_gen = x_generator
            else:
                o_generator = create_generator(o_prompt)
                current_gen = o_generator
            output = await anext(current_gen)

        if isinstance(output, toololo.types.ToolResult):
            if output.func == game.make_move and output.success:
                print("\nCurrent board:")
                game.print_board()

    print("\n=== Game Over ===")
    game.print_board()

    winner = game.get_winner()
    if winner:
        print(f"Player {winner} wins!")
    else:
        print("It's a draw!")
