from nba_api.stats.endpoints import leaguegamelog, playergamelogs, boxscoretraditionalv3
from nba_api.stats.static import players


def main() -> None:
    # Search for players.
    found_player = players.find_players_by_last_name("knecht")

    lebron_id = "2544"
    jokic_id = "203999"
    hayes_id = "1629637"

    season = "2024-25"

    # League GameLogs.
    # league_gamelog = leaguegamelog.LeagueGameLog(season=season).get_data_frames()[0]
    # league_gamelog.to_csv("league_gamelog_validation_v2.csv", index=False)

    # Player GameLogs.
    # player_gamelog = playergamelogs.PlayerGameLogs(player_id_nullable=hayes_id, season_nullable=season).get_data_frames()[0]
    # player_gamelog.to_csv("player_gamelog_validation_v2.csv", index=False)

    # BoxScores.
    boxscore = boxscoretraditionalv3.BoxScoreTraditionalV3(game_id="").get_data_frames()[0]

    return


if __name__ == "__main__":
    main()
