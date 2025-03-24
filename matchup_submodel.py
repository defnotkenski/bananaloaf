import sys
from typing import Literal
import sklearn.calibration
from nba_api.stats.static import players
from pathlib import Path
import polars
from sklearn.model_selection import LeaveOneOut, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
import xgboost
from tqdm import tqdm
import optuna
import multiprocessing

debug_print = f"🍯 [DEBUG] -- "


class PlayerMatchUp:
    def __init__(self, player_last_arg: str, player_gamelogs_csv_arg: Path, league_gamelogs_csv_arg: Path):
        self.filtered_opp_gamelogs: polars.DataFrame | None = None

        player_gamelog_raw: polars.DataFrame = polars.read_csv(player_gamelogs_csv_arg)
        league_gamelog_raw: polars.DataFrame = polars.read_csv(league_gamelogs_csv_arg)

        self.player_gamelogs_df: polars.DataFrame = self._cleanup_df(player_gamelog_raw, is_iso_8601=True)
        self.league_gamelogs_df: polars.DataFrame = self._cleanup_df(league_gamelog_raw, is_iso_8601=False)

        self.player_last: str = player_last_arg

    @staticmethod
    def _calc_team_possessions(game_stats: dict) -> float:

        possessions = 0.96 * (game_stats["FGA"] + 0.44 * game_stats["FTA"] - game_stats["OREB"] + game_stats["TOV"])

        return possessions

    @staticmethod
    def _cleanup_df(polars_df: polars.DataFrame, is_iso_8601: bool) -> polars.DataFrame:
        if is_iso_8601:
            convert_dates = polars_df.with_columns(polars.col("GAME_DATE").cast(polars.Datetime))
            return convert_dates

        convert_dates = polars_df.with_columns(polars.col("GAME_DATE").str.to_datetime("%Y-%m-%d"))
        return convert_dates

    def _get_opp_other_opp_boxscore(self, opp_team_id_arg: int, opp_game_id_arg: int) -> polars.DataFrame:
        opp_other_opp_boxscore = self.league_gamelogs_df.filter(
            (polars.col("GAME_ID") == opp_game_id_arg) & (polars.col("TEAM_ID") != opp_team_id_arg)
        )

        return opp_other_opp_boxscore

    def _get_player_info(self) -> str:
        # This is currently set up to return the player id.
        get_player_id_by_name = players.find_players_by_last_name(self.player_last)

        if len(get_player_id_by_name) > 1:
            print(f"{debug_print} There may be multiple players with that last name. Exiting to be safe...")
            sys.exit()

        print(f"{debug_print} Found {get_player_id_by_name[0]['full_name']}. Player ID is {get_player_id_by_name[0]['id']}")

        player_id: str = str(get_player_id_by_name[0]["id"])

        return player_id

    def _calc_opp_def_rating(self, filtered_opp_gamelogs: polars.DataFrame) -> float:
        master_opp_posessions_allowed = []
        master_opp_points_allowed = []

        for opp_game in filtered_opp_gamelogs.iter_rows(named=True):
            opp_possessions_allowed = self._calc_team_possessions(opp_game)
            opp_points_allowed = opp_game["PTS"]

            master_opp_posessions_allowed.append(opp_possessions_allowed)
            master_opp_points_allowed.append(opp_points_allowed)

        sum_master_opp_posessions_allowed = sum(master_opp_posessions_allowed)
        sum_master_opp_points_allowed = sum(master_opp_points_allowed)

        opp_defensive_rating = (sum_master_opp_points_allowed / sum_master_opp_posessions_allowed) * 100

        return opp_defensive_rating

    def _calc_average_opp_pace(self, filtered_opp_gamelogs: polars.DataFrame) -> float:
        master_opp_pace = []

        # Iterate through recent games from the opponent gamelog.
        for opp_game in filtered_opp_gamelogs.iter_rows(named=True):
            opp_game_id = opp_game["GAME_ID"]
            opp_team_id = opp_game["TEAM_ID"]

            # Calculate the possessions for the opponent.
            opp_possessions = self._calc_team_possessions(opp_game)

            # Calculate the possessions for the opponent's other opponent.
            opp_other_opp = self.league_gamelogs_df.filter((polars.col("GAME_ID") == opp_game_id) & (polars.col("TEAM_ID") != opp_team_id))
            opp_other_opp_possessions = self._calc_team_possessions(opp_other_opp.row(0, named=True))

            game_pace = (opp_possessions + opp_other_opp_possessions) / 2

            master_opp_pace.append(game_pace)

        # Calculate the average pace across recent games.
        average_opp_pace = sum(master_opp_pace) / len(master_opp_pace)

        return average_opp_pace

    def _calc_avg_opp_points_allowed(self, filtered_opp_gamelogs: polars.DataFrame) -> float:
        master_opp_pts_allowed = []

        for opp_game in filtered_opp_gamelogs.head(5).iter_rows(named=True):
            opp_game_id = opp_game["GAME_ID"]
            opp_team_id = opp_game["TEAM_ID"]

            opp_other_opp_boxscore = self.league_gamelogs_df.filter((polars.col("GAME_ID") == opp_game_id) & (polars.col("TEAM_ID") != opp_team_id))
            opp_pts_allowed = opp_other_opp_boxscore["PTS"].item()

            master_opp_pts_allowed.append(opp_pts_allowed)

        avg_opp_pts_allowed = sum(master_opp_pts_allowed) / len(master_opp_pts_allowed)

        return avg_opp_pts_allowed

    def _calc_opp_efg_perc(self) -> float:
        master_fgm, master_3pm, master_fga = [], [], []

        for opp_game in self.filtered_opp_gamelogs.head(5).iter_rows(named=True):
            opp_game_id = opp_game["GAME_ID"]
            opp_team_id = opp_game["TEAM_ID"]

            opp_other_opp_boxscore = self._get_opp_other_opp_boxscore(opp_game_id_arg=opp_game_id, opp_team_id_arg=opp_team_id)
            opp_fg_made = opp_other_opp_boxscore["FGM"].item()
            opp_3p_made = opp_other_opp_boxscore["FG3M"].item()
            opp_fg_attempted = opp_other_opp_boxscore["FGA"].item()

            master_fgm.append(opp_fg_made)
            master_3pm.append(opp_3p_made)
            master_fga.append(opp_fg_attempted)

        opp_efg_perc_allowed = (sum(master_fgm) + 0.5 * sum(master_3pm)) / sum(master_fga)

        return opp_efg_perc_allowed

    @staticmethod
    def _train_xgboost(train_data_arg: list[dict]) -> None:
        outer_loocv = LeaveOneOut()

        # ----- Prepare polars dataframe. -----
        train_data_df = polars.DataFrame(train_data_arg)

        x = train_data_df.drop(["game_date", "target"]).to_numpy()
        y = train_data_df.select("target").to_numpy().ravel()

        # We'll store these to track performance across the outer loop
        outer_y_true = []
        outer_y_pred = []
        outer_y_pred_prob = []

        n_jobs = multiprocessing.cpu_count()
        print(f"🐝 Number of cores to be used for hyperparameter sweep: {n_jobs}")

        # Outer loop
        for idx, (train_idx, test_idx) in enumerate(outer_loocv.split(x)):
            print(f"🐝 Current iteration: {idx + 1}")

            x_train, x_test = x[train_idx], x[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Inner LOOCV for hyperparam search
            inner_loocv = LeaveOneOut()

            def optuna_objective(trial: optuna.Trial) -> float:

                max_depth = trial.suggest_int("max_depth", 2, 5)
                learning_rate = trial.suggest_categorical("learning_rate", [0.01, 0.02, 0.05, 0.1])
                n_estimators = trial.suggest_categorical("n_estimators", [50, 100, 200, 300, 500])
                subsample = trial.suggest_categorical("subsample", [0.6, 0.8, 1.0])
                colsample_bytree = trial.suggest_categorical("colsample_bytree", [0.6, 0.8, 1.0])
                colsample_bylevel = trial.suggest_categorical("colsample_bylevel", [0.6, 1.0])
                colsample_bynode = trial.suggest_categorical("colsample_bynode", [0.6, 1.0])
                min_child_weight = trial.suggest_int("min_child_weight", 1, 10)
                gamma = trial.suggest_categorical("gamma", [0, 0.1, 0.5, 1])
                reg_alpha = trial.suggest_categorical("reg_alpha", [0, 0.001, 0.01, 0.1, 1, 10])
                reg_lambda = trial.suggest_categorical("reg_lambda", [0.1, 1.0, 2.0, 5.0, 10])
                scale_pos_weight = trial.suggest_categorical("scale_pos_weight", [1, 2, 3, 5])
                booster = trial.suggest_categorical("booster", ["gbtree", "dart"])

                optuna_objective_model = xgboost.XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    verbosity=0,
                    # ----- CUDA support. -----
                    # device="cuda",
                    # tree_method="hist",
                    # sampling_method=sampling_method,
                    # ----- Param sweep. -----
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    n_estimators=n_estimators,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    colsample_bylevel=colsample_bylevel,
                    colsample_bynode=colsample_bynode,
                    min_child_weight=min_child_weight,
                    gamma=gamma,
                    reg_alpha=reg_alpha,
                    reg_lambda=reg_lambda,
                    scale_pos_weight=scale_pos_weight,
                    booster=booster,
                )

                scores = cross_val_score(estimator=optuna_objective_model, X=x, y=y, cv=inner_loocv, scoring="accuracy", n_jobs=-1)

                return scores.mean()

            optuna.logging.set_verbosity(optuna.logging.WARNING)
            progress_bar = tqdm(total=250, desc="Optuna Trials")

            def update_pbar(_study, _trial):
                progress_bar.update(1)

            optuna_sampler = optuna.samplers.TPESampler()
            _pruner = optuna.pruners.HyperbandPruner()

            study = optuna.create_study(sampler=optuna_sampler, pruner=None, direction="maximize")
            study.optimize(optuna_objective, n_jobs=n_jobs, n_trials=250, show_progress_bar=False, callbacks=[update_pbar])

            progress_bar.close()

            best_params = study.best_params
            best_accuracy_found = study.best_value

            print(f"🐝 Optuna best accuracy (LOOCV estimate): {best_accuracy_found:.4f}")
            # print(f"🐝 Optuna best params: {best_params}")

            model = xgboost.XGBClassifier(**best_params, objective="binary:logistic", eval_metric="logloss", verbosity=0)
            model.fit(x_train, y_train)

            y_pred_test = model.predict(x_test)[0]
            y_pred_prob_test = model.predict_proba(x_test)[0, 1]

            outer_y_true.append(y_test[0])
            outer_y_pred.append(y_pred_test)
            outer_y_pred_prob.append(y_pred_prob_test)

        # ----- Evaluate overall performance across the outer folds. -----
        final_accuracy = accuracy_score(outer_y_true, outer_y_pred)
        print(f"🐝 Nested LOOCV final accuracy: {final_accuracy:.4f}")

        # ----- Calculate confifence from predictions. -----
        results_converted = []

        for true_class, pred, prob in zip(outer_y_true, outer_y_pred, outer_y_pred_prob):
            confidence = abs(prob - 0.5) * 2

            results_converted.append(
                {
                    "true_class": true_class,
                    "predicted_class": pred,
                    "prob_class_1": prob,
                    "confidence": confidence,
                    "correct": pred == true_class,
                }
            )

        results_converted_df = polars.DataFrame(results_converted)

        # ----- Evaluate high confidence predictions. -----
        confidence_threshold = 0.60
        high_confidence_df = results_converted_df.filter(polars.col("confidence") >= confidence_threshold)

        accuracy_high_confidence = high_confidence_df.get_column("correct").mean()

        print(f"🐝 Accuracy of high-confidence predictions: {accuracy_high_confidence}")
        print(high_confidence_df)

        return

    def train_matchup_context_submodel(self, bet_line_under_or_over: Literal["under", "over"], points_or_equiv: int) -> None:
        """
        This function is for training (not predicting) the match-up and context submodel only.
        ---
        Development Scenario:
        You are betting on whether or not Nikola Jokic will be over 20 points in his next upcoming game.

        Feature Set:
        - Opponent's Defensive Rating.

        """
        # Get the defensive rating for each game before the target game.
        feature_extraction: list[dict] = []

        # Grab the historical games for the player to extract features and train submodel.

        # Iterate through all hsitorical games the player has played except for the last 5 games.
        for player_game in self.player_gamelogs_df.head(-5).iter_rows(named=True):
            # This is the current game that we are extracting the features from.

            player_game_id = player_game["GAME_ID"]
            player_team_id = player_game["TEAM_ID"]
            game_date = player_game["GAME_DATE"]

            # Get the opponent's past games.
            game_boxscores = self.league_gamelogs_df.filter(polars.col("GAME_ID") == player_game_id)

            opp_boxscore = game_boxscores.filter(polars.col("TEAM_ID") != player_team_id)

            opp_team_id = opp_boxscore["TEAM_ID"].item()

            opp_team_gamelogs = self.league_gamelogs_df.filter(polars.col("TEAM_ID") == opp_team_id)

            # Only return the games up to the game we are predicting.
            filtered_opp_team_gamelogs: polars.DataFrame = opp_team_gamelogs.filter(polars.col("GAME_DATE") < game_date)
            self.filtered_opp_gamelogs = filtered_opp_team_gamelogs

            # Calculate the opponent defensive rating (window = season long).
            opp_def_rating = self._calc_opp_def_rating(filtered_opp_gamelogs=filtered_opp_team_gamelogs)

            # Calculate average opponent pace (window = season long).
            avg_opp_pace = self._calc_average_opp_pace(filtered_opp_gamelogs=filtered_opp_team_gamelogs)

            # Calculate average opponent points allowed (window = last 5 games).
            avg_opp_pts_allowed = self._calc_avg_opp_points_allowed(filtered_opp_gamelogs=filtered_opp_team_gamelogs)

            # Calculate the opponent effective field goal percentage (window = last 5 games).
            opp_efg_perc_allowed = self._calc_opp_efg_perc()

            # Whether or not the player was under or over their betting line.
            pts_scored = player_game["PTS"]
            prop_line_outcome = None

            if bet_line_under_or_over == "under":
                prop_line_outcome = pts_scored < points_or_equiv

            if bet_line_under_or_over == "over":
                prop_line_outcome = pts_scored > points_or_equiv

            # Add player game features to master list.
            feature_extraction.append(
                {
                    "game_date": game_date,
                    "opp_def_rating": opp_def_rating,
                    "opp_avg_pace": avg_opp_pace,
                    "avg_opp_pts_allowed": avg_opp_pts_allowed,
                    "opp_efg_perc_allowed": opp_efg_perc_allowed,
                    "target": 1 if prop_line_outcome else 0,
                }
            )

        self._train_xgboost(train_data_arg=feature_extraction)

        return


if __name__ == "__main__":
    # Parameters.
    player_last: str = "jokic"
    season: str = "2024-25"

    curr_dir = Path.cwd()

    player_gamelogs_csv = curr_dir.joinpath("player_gamelog_validation_v2.csv")
    league_gamelogs_csv = curr_dir.joinpath("league_gamelog_validation_v2.csv")

    player_matchup_context = PlayerMatchUp(
        player_last_arg=player_last, player_gamelogs_csv_arg=player_gamelogs_csv, league_gamelogs_csv_arg=league_gamelogs_csv
    )

    player_matchup_context.train_matchup_context_submodel(bet_line_under_or_over="over", points_or_equiv=6)
