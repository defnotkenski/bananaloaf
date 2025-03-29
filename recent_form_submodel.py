from typing import Literal
import polars
from lecommons import LeCommons, DEBUG_PRINT
from letrainer_utils import LeTrainerUtils

debug_print = DEBUG_PRINT


class RecentFormSubModel:
    def __init__(self, dataframe: polars.DataFrame, over_under: Literal["over", "under"], pts: int):
        self.dataframe = dataframe
        self.over_under = over_under
        self.pts = pts

    @staticmethod
    def _get_average_pts_and_std(dataframe: polars.DataFrame) -> tuple[float, float]:
        avg_games_pts = dataframe.get_column("PTS").mean()
        games_std = dataframe.get_column("PTS").std(ddof=1)

        return avg_games_pts, games_std

    @staticmethod
    def _get_avg_mins(dataframe: polars.DataFrame) -> float:
        avg_mins = dataframe.get_column("MIN").mean()

        return avg_mins

    @staticmethod
    def _get_fg_perc(dataframe: polars.DataFrame) -> float:
        fga = dataframe.get_column("FGA").sum()
        fgm = dataframe.get_column("FGM").sum()

        fg_perc = fgm / fga

        return fg_perc

    @staticmethod
    def _get_avg_tov(dataframe: polars.DataFrame) -> float:
        avg_tov = dataframe.get_column("TOV").mean()

        return avg_tov

    @staticmethod
    def train_model(master_features_arg: list[dict]) -> None:
        letrainer = LeTrainerUtils()
        scores, scores_mean, scores_std = letrainer.train_xgboost(feature_extractions_arg=master_features_arg, n_iter=50)

        print(f"\n🍯 ===== FINAL PERFORMANCE ===== 🍯\n")
        print(f"Outer fold accuracy scores:", scores)
        print(f"Mean across outer folds acuracy:", scores_mean)
        print(f"Standard deviation mean across outer folds:", scores_std)

        return

    def prep_data(self) -> list:
        master_features = []
        n_games = -5

        # ===== CALCULATIONS. =====

        for idx, player_game in enumerate(self.dataframe.tail(n_games).iter_rows(named=True)):
            game_date = player_game["GAME_DATE"]
            prior_rows = self.dataframe.filter(polars.col("GAME_DATE") < game_date).tail(5)

            avg_pts, game_std = self._get_average_pts_and_std(dataframe=prior_rows)
            avg_mins = self._get_avg_mins(dataframe=prior_rows)
            avg_tov = self._get_avg_tov(dataframe=prior_rows)
            fg_perc = self._get_fg_perc(dataframe=prior_rows)

            game_pts = player_game["PTS"]
            prop_line_outcome = False

            if self.over_under == "over":
                prop_line_outcome = game_pts > self.pts

            if self.over_under == "under":
                prop_line_outcome = game_pts < self.pts

            # ===== FORMAT DATA FOR TRAINING. =====

            master_features.append(
                {
                    "game_date": player_game["GAME_DATE"],
                    "avg_pts": avg_pts,
                    "game_std": game_std,
                    "avg_mins": avg_mins,
                    "avg_tov": avg_tov,
                    "fg_perc": fg_perc,
                    "target": int(prop_line_outcome),
                }
            )

        return master_features


if __name__ == "__main__":
    player_instance = LeCommons.consolidate_csv(csv_dir="player_gamelogs", is_iso8601=True)
    player_gamelogs = player_instance.filter_by_player(player_name="austin reaves")

    if not player_instance.check_player_continuity(player_df=player_gamelogs):
        raise ValueError(f"{debug_print} There is an issue with player continuity.")

    model_instance = RecentFormSubModel(dataframe=player_gamelogs, over_under="over", pts=13)

    master_features = model_instance.prep_data()
    model_instance.train_model(master_features_arg=master_features)
