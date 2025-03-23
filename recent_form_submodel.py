import pandas
import numpy

debug_print = f"🍯🐝---[DEBUG]"


def player_recent_form_submodel() -> None:
    career_df = pandas.read_csv("")

    # pandas.set_option("display.max_columns", None)
    # print(career_df.info())

    filtered = career_df.loc[0:4, ["GAME_ID", "PLAYER_NAME", "GAME_DATE", "MATCHUP", "PTS", "MIN", "TOV", "WL"]]
    print(filtered)

    # Average points and std. dev of last 5.
    last5_points = career_df.loc[0:4, "PTS"].tolist()

    avg_points = numpy.mean(last5_points)
    print(f"{debug_print} The average points over the last 5 games is: {avg_points}")

    std_points = numpy.std(last5_points, ddof=1)
    print(f"{debug_print} The standard deviation of points over the last 5 games is: {std_points}")

    # Average minutes of last 5.
    last5_mins = career_df.loc[0:4, "MIN"].tolist()

    avg_mins = numpy.mean(last5_mins)
    print(f"{debug_print} The avg. minutes over the last 5 games is: {avg_mins:.2f}")

    # Field goal % of last 5.
    last5_fga = sum(career_df.loc[0:4, "FGA"].tolist())
    last5_fgm = sum(career_df.loc[0:4, "FGM"].tolist())
    fg_perc = last5_fgm / last5_fga

    print(f"{debug_print} The FG % over the last 5 games is: {fg_perc:.2f}")

    # Turnovers per game over last 5 games.
    last5_turnovers = career_df.loc[0:4, "TOV"].tolist()
    avg_turnovers = numpy.mean(last5_turnovers)

    print(f"{debug_print} The avg. turnovers over the last 5 games is: {avg_turnovers:.2f}")

    return


if __name__ == "__main__":
    player_recent_form_submodel()
