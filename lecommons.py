import json
import polars
from datetime import datetime
from nba_api.stats.static import players

DEBUG_PRINT = "🍯 [BANANA LOAF][DEBUG]"


class LeCommons:
    def __init__(self, dataframe: polars.DataFrame):
        self.dataframe = dataframe

    @staticmethod
    def _is_iso_8601(dataframe: polars.DataFrame, col_name: str) -> bool:
        first_item = dataframe.head(1)[col_name].item()

        try:
            datetime.fromisoformat(first_item)
            return True
        except ValueError:
            return False

    @classmethod
    def consolidate_csv(cls, csv_files: list[str], is_iso8601: bool):
        master_df = []

        for csv in csv_files:
            csv_to_dataframe = polars.read_csv(csv)

            # ===== CONVERT THE GAME_DATE COLUMN INTO DATETIME FOR EASIER MANIPULATION LATER. =====

            if is_iso8601:
                csv_to_dataframe = csv_to_dataframe.with_columns(polars.col("GAME_DATE").cast(polars.Datetime))

            if not is_iso8601:
                csv_to_dataframe = csv_to_dataframe.with_columns(polars.col("GAME_DATE").str.to_datetime("%Y-%m-%d"))

            master_df.append(csv_to_dataframe)

        combined_df = polars.concat(master_df).sort("GAME_DATE")

        return cls(combined_df)

    def filter_by_player(self, player_name: str) -> polars.DataFrame:
        players_found = players.find_players_by_full_name(player_name)

        if len(players_found) == 0:
            raise ValueError(f"{DEBUG_PRINT} There are no players with that name.")

        if len(players_found) > 1:
            raise ValueError(f"{DEBUG_PRINT} There are multiple players:\n\n{json.dumps(players_found, indent=4)}")

        print(f"Found player: {players_found[0]}")
        player_id = players_found[0]["id"]

        filtered_df = self.dataframe.filter(polars.col("PLAYER_ID") == player_id)

        return filtered_df

    @staticmethod
    def check_player_continuity(player_df: polars.DataFrame) -> bool:
        recent_team = player_df.row(-1, named=True)["TEAM_NAME"]

        for game in player_df.iter_rows(named=True):
            game_team = game["TEAM_NAME"]

            if recent_team != game_team:
                return False

        return True


if __name__ == "__main__":
    player_csv = ["player_gamelogs_s21.csv", "player_gamelogs_s22.csv", "player_gamelogs_s23.csv", "player_gamelogs_s24.csv"]
    league_csv = ["league_gamelogs_s21.csv", "league_gamelogs_s22.csv", "league_gamelogs_s23.csv", "league_gamelogs_s24.csv"]

    players_df = LeCommons.consolidate_csv(csv_files=player_csv, is_iso8601=True)
    league_df = LeCommons.consolidate_csv(csv_files=league_csv, is_iso8601=False)

    player_gamelog = players_df.filter_by_player(player_name="austin reaves")
    is_player_gamelog_healthy = players_df.check_player_continuity(player_df=player_gamelog)

    print(f"-----")
