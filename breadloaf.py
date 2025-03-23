import json
import sys
import textwrap
import time
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import TypedDict
import pyperclip
from openai import OpenAI
from dotenv import load_dotenv
import argparse
import os
import tiktoken

load_dotenv()
script_dir = Path.cwd()
client = OpenAI(api_key=os.environ["PERPLEXITY_API_KEY"], base_url="https://api.perplexity.ai")


class PlayerData(TypedDict):
    player_name: str
    text: str


def load_files(args: Namespace) -> list[PlayerData]:
    dev_json_file = "development.json"
    prod_json_file = "player_data.json"

    if args.dev:
        json_file = dev_json_file
    else:
        json_file = prod_json_file

    with open(script_dir.joinpath(json_file), "r") as read_json:
        player_data = json.load(read_json)

        # print(player_data)
        # print(len(player_data["players_in_question"]))

    return player_data


def setup_parser() -> ArgumentParser:
    # = = = = = =
    # Set up the argument parser so I can run this sh in the terminal.
    # = = = = = =

    # Create the main parser.
    parser = argparse.ArgumentParser(description="A Simple Sports Betting Automation CLI")

    # Create the subparser object.
    subparsers = parser.add_subparsers(dest="command")

    # Subparser for Perplexity Deep Research command.
    parser_deep_research = subparsers.add_parser("research", help="Perform Deep Research on players.")

    parser_deep_research.add_argument("output_dir", default=None, help="Path to output completions.")
    parser_deep_research.add_argument("game_date", default=None, help="Date of the game in question.")
    parser_deep_research.add_argument("--dev", action="store_true", help="Whether or not to run in development mode.")

    # Subparser for consolidating player research.
    parser_player_merge = subparsers.add_parser("merge", help="Merge all player Deep Research.")

    parser_player_merge.add_argument("openai_dir", help="Path to directory containing OpenAI Deep Research Files.")
    parser_player_merge.add_argument("perplexity_dir", help="Path to directory containing Perplexity Deep Research Files.")
    parser_player_merge.add_argument("--copy", action="store_true", help="Copy merged research to clipboard.")
    parser_player_merge.add_argument("--players", type=int, required=True, help="Number of researched NBA players in question.")
    parser_player_merge.add_argument("--top_n", type=int, required=True, help="Limit the ranking to a certain amount of players.")
    parser_player_merge.add_argument("--write", type=str, required=False, help="File path to write the merged research into a file.")

    return parser


def format_time_remaining(seconds: float) -> str:
    if seconds >= 3600:
        # If time remaining is over 1 hour.
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)

        return f"{hours} hrs {minutes} mins"
    elif seconds >= 60:
        # If time remaining is under 1 hour.
        minutes = int(seconds / 60)

        return f"{minutes} mins"

    return "Could not format time."


def merge_all_deep_research_files(args: Namespace) -> str:
    openai_deep_research_dump: list[str] = []
    perplexity_deep_research_dump: list[str] = []

    openai_deep_research_dir: Path = Path(args.openai_dir)
    perplexity_deep_research_dir: Path = Path(args.perplexity_dir)

    number_players = args.players
    rank_limit = args.top_n

    prompt = textwrap.dedent(
        f"""\
    # Task Description
    I've provided detailed research from two separate researchers - Researcher A and Researcher B - on {number_players} NBA players. The research predicts whether each player will go **over or under** a specified betting line. Each researcher’s findings may or may not agree with the other's analysis, leading to potential contradictions.
    
    ## Instructions:
    1. Carefully analyze and synthesize both provided researchers’ analyses.
    2. Combine these insights with your analytical reasoning to identify and rank the top {rank_limit} players based on the highest statistical probability (confidence) of their predicted betting outcomes being correct.
    3. Only select players explicitly included in this research.
    4. Provide a brief explanation for each player's ranking and confidence level, clearly noting insights derived from your own analysis as well as the provided data.
    
    ## Research Data
    
    """
    )

    # Iterate through each txt file in the directory and dump all contents into a list.
    for openai_file in os.listdir(openai_deep_research_dir):
        if openai_file.endswith(".txt"):
            with open(openai_deep_research_dir.joinpath(openai_file), "r") as read_openai_file:
                openai_deep_research_dump.append(read_openai_file.read())

    # Combine all items into a singular string seperated by double linebreaks.
    merged_openai_dump = "\n\n".join(openai_deep_research_dump)

    # Iterate through each txt file in the directory and dump all contents into a list.
    for file in os.listdir(perplexity_deep_research_dir):
        if file.endswith(".txt"):
            with open(perplexity_deep_research_dir.joinpath(file), "r") as read_file:
                perplexity_deep_research_dump.append(read_file.read())

    # Combine all items into a singular string seperated by double linebreaks.
    merged_perplexity_dump = "\n\n".join(perplexity_deep_research_dump)

    # Format everything to include the prompt and all research.
    openai_add_heading = "### Researcher A:\n\n" + merged_openai_dump
    perplexity_add_heading = "### Researcher B:\n\n" + merged_perplexity_dump
    merged_both_research = openai_add_heading + "\n\n---\n\n" + perplexity_add_heading

    add_prompt = prompt + merged_both_research

    # Count the number of tokens.
    enc = tiktoken.encoding_for_model("o1")
    assert enc.decode(enc.encode(add_prompt)) == add_prompt
    token_count = len(enc.encode(add_prompt))

    print(f"🍯🐝---[DEBUG]: Token Count: {token_count}")

    return add_prompt


def run_perplexity_deep_research(args: Namespace, players_in_question: list[PlayerData]) -> None:
    print(f"🦖 [DEBUG]: Running Perplexity Deep Research for {len(players_in_question)} players.\n")

    output_dir_path: Path = Path(args.output_dir)
    times_taken: list = []

    # Modify each item to include prompt for Perplexity.

    game_date: str = args.game_date
    add_prompt: str = (
        f"With that information, I need you to find whether or not that NBA player will be over or under their stat line for their game vs their opponent on {game_date}. I need a high degree of statistical accuracy with your final, definitive answer. Take into account factors such as high-variance players, back-to-back games and fatigue, injury and rotation news, pace and defensive matchups, blowout risk, projections, and team trends as well as any other factors or statistical models or algorithms that is relevant in achieving the most accurate answer. Include the statistical probability (confidence) as a percentage for your predicted outcome."
    )

    modified_players_in_question: list[PlayerData] = [
        {"player_name": player["player_name"], "text": player["text"] + " " + add_prompt} for player in players_in_question
    ]

    # Run Perplexity Deep Research for each player in list.

    for index, player in enumerate(modified_players_in_question):
        print(f"🦖 [DEBUG]: Running Deep Research for player: {player['player_name']} - {index + 1} of {len(modified_players_in_question)}")

        request_start = time.time()

        completion = client.chat.completions.create(
            model="sonar-deep-research",
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": player["text"]}],
                },
            ],
        )

        # Write the completion output to file at path from cli args.

        prepend: str = f"**Research for {player['player_name']}**\n\n"
        prepend_output: str = prepend + completion.choices[0].message.content

        with open(output_dir_path.joinpath(f"{player['player_name'].lower().replace(' ', '_')}.txt"), "w") as write_file:
            write_file.write(prepend_output)

        # Track time taken for request.

        request_end = time.time()
        elapsed_time = request_end - request_start
        times_taken.append(elapsed_time)

        # Compute estimated remaining time.

        avg_time_per_request = sum(times_taken) / len(times_taken)
        remaining_requests = len(modified_players_in_question) - (index + 1)
        estimated_time_left = avg_time_per_request * remaining_requests

        formatted_time = format_time_remaining(seconds=estimated_time_left)

        print(f"🦖 [DEBUG]: Running Deep Research for player: {player['player_name']} - {index + 1} of {len(modified_players_in_question)} DONE!\n")
        print(f"🦖 [DEBUG]: Estimated time remaining: {formatted_time}\n")


if __name__ == "__main__":
    # = = = = = =
    # Set up the argument parser so I can run this sh in the terminal & any other things.
    # = = = = = =

    bet_parser = setup_parser()
    bet_args = bet_parser.parse_args()

    # = = = = = =
    # Run player merge on Deep Research output.
    # = = = = = =

    if bet_args.command == "merge":
        print(f"🍯🐝---[DEBUG]: Running Deep Research merge for all files in: {bet_args.openai_dir} & {bet_args.perplexity_dir}")

        merged_research = merge_all_deep_research_files(args=bet_args)

        if bet_args.write:
            write_path = Path(bet_args.write)

            with open(write_path, "w") as write_prompt:
                write_prompt.write(merged_research)

            print(f"🍯🐝---[DEBUG]: Successfully written to file: {write_path}")

        if bet_args.copy:
            pyperclip.copy(merged_research)
            print(f"🍯🐝---[DEBUG]: Copied merged research to clipboard!")

        sys.exit()

    # = = = = = =
    # Run Perplexity Deep Research on each of the players.
    # = = = = = =

    players_in_question_list = load_files(args=bet_args)

    run_perplexity_deep_research(args=bet_args, players_in_question=players_in_question_list)

    print(f"🦖 [DEBUG]: Perplexity Deep Research COMPLETED! 👏🏼")
