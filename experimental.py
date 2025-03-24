from nba_api.stats.endpoints import leaguegamelog, playergamelogs, boxscoretraditionalv3
from nba_api.stats.static import players
import polars


def main() -> None:
    # Search for players.
    found_player = players.find_players_by_last_name("knecht")

    lebron_id = "2544"
    jokic_id = "203999"
    hayes_id = "1629637"

    season = "2023-24"

    # League GameLogs.
    # league_gamelog = leaguegamelog.LeagueGameLog(season=season).get_data_frames()[0]
    # league_gamelog.to_csv("league_gamelog_validation_v2_extension.csv", index=False)

    # Player GameLogs.
    # player_gamelog = playergamelogs.PlayerGameLogs(player_id_nullable=hayes_id, season_nullable=season).get_data_frames()[0]
    # player_gamelog.to_csv("player_gamelog_validation_v2_extension.csv", index=False)

    # -----

    player_df = polars.read_csv("player_gamelog_validation_v2.csv")
    player_df_ext = polars.read_csv("player_gamelog_validation_v2_extension.csv")

    player_concat = polars.concat([player_df, player_df_ext])
    # player_concat.write_csv("player_gamelog_concat_v1.csv")

    league_df = polars.read_csv("league_gamelog_validation_v2.csv")
    league_df_ext = polars.read_csv("league_gamelog_validation_v2_extension.csv")

    league_concat = polars.concat([league_df_ext, league_df])
    league_concat.write_csv("league_gamelog_concat_v1.csv")

    return


if __name__ == "__main__":
    main()
