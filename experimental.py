from nba_api.stats.endpoints import leaguegamelog, playergamelogs
from nba_api.stats.static import players


def main() -> None:
    """
    Cache PlayerGameLogs for 3 years prior and request the current season dataframe once per betting period.

    This way, you can query the same dataset for every player in question.

    :return: None
    """
    # ----- Search for players. -----

    # has_player_stayed("knecht")
    player_id = players.find_players_by_last_name("davis")

    lebron_id = "2544"
    jokic_id = "203999"
    hayes_id = "1629637"

    # -----

    # League GameLogs.
    # league_gamelog = leaguegamelog.LeagueGameLog(season="2024-25").get_data_frames()[0]
    # league_gamelog.to_csv("league_gamelogs_s24.csv", index=False)

    # Player GameLogs.
    dev_nba_api = playergamelogs.PlayerGameLogs(season_nullable="2024-25").get_data_frames()[0]
    dev_nba_api.to_csv("player_gamelogs_s24.csv", index=False)

    # -----

    # player_df = polars.read_csv("player_gamelog.csv")
    # player_df_ext = polars.read_csv("player_gamelog.csv")
    #
    # player_concat = polars.concat([player_df, player_df_ext])
    # # player_concat.write_csv("player_gamelog_concat_v1.csv")
    #
    # league_df = polars.read_csv("league_gamelog.csv")
    # league_df_ext = polars.read_csv("league_gamelog.csv")
    #
    # league_concat = polars.concat([league_df_ext, league_df])
    # league_concat.write_csv("league_gamelog_concat_v1.csv")

    return


if __name__ == "__main__":
    main()
