from typing import Literal
import numpy
import polars
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
import xgboost
from tqdm import tqdm
import optuna
import multiprocessing
from lecommons import LeCommons, DEBUG_PRINT

debug_print = DEBUG_PRINT


class PlayerMatchUp:
    def __init__(self, player_gamelogs_df_arg: polars.DataFrame, league_gamelogs_df_arg: polars.DataFrame):
        self.filtered_opp_gamelogs: polars.DataFrame | None = None

        self.player_gamelogs_df: polars.DataFrame = player_gamelogs_df_arg
        self.league_gamelogs_df: polars.DataFrame = league_gamelogs_df_arg

    @staticmethod
    def _calc_team_possessions(game_stats: dict) -> float:

        possessions = 0.96 * (game_stats["FGA"] + 0.44 * game_stats["FTA"] - game_stats["OREB"] + game_stats["TOV"])

        return possessions

    def _get_opp_other_opp_boxscore(self, opp_team_id_arg: int, opp_game_id_arg: int) -> polars.DataFrame:
        opp_other_opp_boxscore = self.league_gamelogs_df.filter(
            (polars.col("GAME_ID") == opp_game_id_arg) & (polars.col("TEAM_ID") != opp_team_id_arg)
        )

        return opp_other_opp_boxscore

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

    # @staticmethod
    # def _train_xgboost(train_data_arg: list[dict], n_iter: int = 100) -> None:
    #     # -----
    #
    #     def update_pbar(_study, _trial):
    #         progress_bar.update(1)
    #
    #     optuna.logging.set_verbosity(optuna.logging.WARNING)
    #
    #     # ===== PREPARE POLARS DATAFRAME. =====
    #
    #     data_df = polars.DataFrame(train_data_arg)
    #     sort_data_df = data_df.sort("game_date", descending=False)
    #
    #     x = sort_data_df.drop(["game_date", "target"]).to_numpy()
    #     y = sort_data_df.get_column("target").to_numpy()
    #
    #     # ===== CHECK PRECENTAGE SPLITS OF TARGET DATA. =====
    #
    #     target_balance_check = (
    #         sort_data_df.get_column("target").value_counts().with_columns((polars.col("count") / polars.sum("count") * 100).alias("percentage"))
    #     )
    #
    #     percentage_1 = target_balance_check.filter(polars.col("target") == 1).to_dict(as_series=False)
    #     percentage_0 = target_balance_check.filter(polars.col("target") == 0).to_dict(as_series=False)
    #
    #     print(f"🐝 There are {percentage_1['count'][0]} with class {percentage_1['target'][0]} which is {percentage_1['percentage'][0]:.2f}")
    #     print(f"🐝 There are {percentage_0['count'][0]} with class {percentage_0['target'][0]} which is {percentage_0['percentage'][0]:.2f}")
    #
    #     # ===== SET UP OUTER LOOP (TIME SERIES). =====
    #
    #     outer_n_splits = 28
    #     outer_test_size = 1
    #
    #     outer_tscv = TimeSeriesSplit(n_splits=outer_n_splits, test_size=outer_test_size)
    #     outer_scores = []
    #     # outer_predict = []
    #     # outer_predict_prob = []
    #
    #     # ===== LOOP OVER OUTER FOLDS. =====
    #
    #     for train_idx, val_idx in outer_tscv.split(x):
    #
    #         progress_bar = tqdm(total=n_iter, desc="Optuna Trials")
    #
    #         # ===== SPLIT TRAIN DATA. =====
    #
    #         x_train_outer, y_train_outer = x[train_idx], y[train_idx]
    #         x_val_outer, y_val_outer = x[val_idx], y[val_idx]
    #
    #         # ===== SET UP INNER LOOP (TIME SERIES). =====
    #
    #         inner_n_splits = 28
    #         inner_test_size = 1
    #
    #         inner_tscv = TimeSeriesSplit(n_splits=inner_n_splits, test_size=inner_test_size)
    #
    #         n_jobs = int(multiprocessing.cpu_count())
    #         print(f"🐝 Number of cores to be used for hyperparameter sweep: {n_jobs}")
    #
    #         # ===== DEFINE OPTUNA OBJECTIVE. =====
    #
    #         def optuna_objective(trial: optuna.Trial) -> numpy.floating:
    #             optuna_scores = []
    #
    #             param_sweep = {
    #                 "max_depth": trial.suggest_int("max_depth", 2, 5),
    #                 "learning_rate": trial.suggest_categorical("learning_rate", [0.01, 0.02, 0.05, 0.1]),
    #                 "n_estimators": trial.suggest_categorical("n_estimators", [50, 100, 200, 300, 500]),
    #                 "subsample": trial.suggest_categorical("subsample", [0.6, 0.8, 1.0]),
    #                 "colsample_bytree": trial.suggest_categorical("colsample_bytree", [0.6, 0.8, 1.0]),
    #                 "colsample_bylevel": trial.suggest_categorical("colsample_bylevel", [0.6, 1.0]),
    #                 "colsample_bynode": trial.suggest_categorical("colsample_bynode", [0.6, 1.0]),
    #                 "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
    #                 "gamma": trial.suggest_categorical("gamma", [0, 0.1, 0.5, 1]),
    #                 "reg_alpha": trial.suggest_categorical("reg_alpha", [0, 0.001, 0.01, 0.1, 1, 10]),
    #                 "reg_lambda": trial.suggest_categorical("reg_lambda", [0.1, 1.0, 2.0, 5.0, 10]),
    #                 "scale_pos_weight": trial.suggest_categorical("scale_pos_weight", [1, 2, 3, 5]),
    #                 # "booster": trial.suggest_categorical("booster", ["gbtree", "dart"]),
    #             }
    #
    #             for inner_train_idx, inner_val_idx in inner_tscv.split(x_train_outer):
    #                 x_train_inner, y_train_inner = x_train_outer[inner_train_idx], y_train_outer[inner_train_idx]
    #                 x_val_inner, y_val_inner = x_train_outer[inner_val_idx], y_train_outer[inner_val_idx]
    #
    #                 optuna_objective_model = xgboost.XGBClassifier(
    #                     objective="binary:logistic",
    #                     eval_metric="logloss",
    #                     verbosity=0,
    #                     # ===== CUDA support. =====
    #                     device="cuda",
    #                     # tree_method="hist",
    #                     # sampling_method=sampling_method,
    #                     # ===== Param sweep. =====
    #                     **param_sweep,
    #                 )
    #
    #                 optuna_objective_model.fit(x_train_inner, y_train_inner)
    #
    #                 # ===== Run predictions and evaluate. =====
    #
    #                 y_predict_inner = optuna_objective_model.predict(x_val_inner)
    #                 accuracy_score_optuna = accuracy_score(y_val_inner, y_predict_inner)
    #
    #                 optuna_scores.append(accuracy_score_optuna)
    #
    #             return numpy.mean(optuna_scores)
    #
    #         # ===== RUN OPTUNA. =====
    #
    #         optuna_sampler = optuna.samplers.TPESampler()
    #         pruner = optuna.pruners.MedianPruner(n_startup_trials=10)
    #
    #         study = optuna.create_study(sampler=optuna_sampler, pruner=pruner, direction="maximize")
    #         study.optimize(optuna_objective, n_jobs=n_jobs, n_trials=n_iter, show_progress_bar=False, callbacks=[update_pbar])
    #
    #         progress_bar.close()
    #
    #         optuna_best_params = study.best_params
    #         optuna_best_accuracy_found = study.best_value
    #
    #         print(f"🐝 Optuna best params accuracy: {optuna_best_accuracy_found:.4f}")
    #         print(f"🐝 Optuna best params: {optuna_best_params}")
    #
    #         # ===== RETRAIN ON OUTERLOOP WITH BEST PARAMS FROM OPTUNA. =====
    #
    #         best_param_model = xgboost.XGBClassifier(**optuna_best_params, objective="binary:logistic", eval_metric="logloss", verbosity=0)
    #         best_param_model.fit(x_train_outer, y_train_outer)
    #
    #         # ===== PREDICT AND EVALUATE. =====
    #
    #         y_pred = best_param_model.predict(x_val_outer)
    #         # y_pred_prob = best_param_model.predict_proba(x_val_outer)[:, 1]
    #
    #         best_param_accuracy = accuracy_score(y_true=y_val_outer, y_pred=y_pred)
    #
    #         outer_scores.append(best_param_accuracy)
    #         # outer_predict.append(y_pred)
    #         # outer_predict_prob.append(y_pred_prob)
    #
    #     # ===== OVERALL PERFORMANCE (AVERAGE ACROSS OUTER FOLDS). =====
    #
    #     print(f"\n🍯 ===== FINAL PERFORMANCE ===== 🍯\n")
    #     print(f"Outer fold accuracy scores:", outer_scores)
    #     print(f"Mean across outer folds acuracy:", numpy.mean(outer_scores))
    #     print(f"Standard deviation mean across outer folds:", numpy.std(outer_scores))
    #
    #     # results_converted = []
    #     #
    #     # for true_class, pred, prob in zip(y, outer_predict, outer_predict_prob):
    #     #     confidence = abs(prob - 0.5) * 2
    #     #
    #     #     results_converted.append(
    #     #         {
    #     #             "true_class": true_class,
    #     #             "predicted_class": pred,
    #     #             "prob_class_1": prob,
    #     #             "confidence": confidence,
    #     #             "correct": pred == true_class,
    #     #         }
    #     #     )
    #     #
    #     # results_converted_df = polars.DataFrame(results_converted)
    #     #
    #     # # ----- Evaluate high confidence predictions. -----
    #     #
    #     # confidence_threshold = 0.70
    #     #
    #     # high_confidence_df = results_converted_df.filter(polars.col("confidence") >= confidence_threshold)
    #     #
    #     # if high_confidence_df.height == 0:
    #     #     print(f"🐝 There were no samples that meet the confidence threshold of {confidence_threshold}")
    #     #     return
    #     #
    #     # accuracy_high_confidence = high_confidence_df.get_column("correct").mean()
    #     #
    #     # print(f"🐝 Accuracy of high-confidence predictions: {accuracy_high_confidence}")
    #     # print(high_confidence_df)
    #
    #     return

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
        for player_game in self.player_gamelogs_df.tail(-5).iter_rows(named=True):
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

    player_gamelogs_instance = LeCommons.consolidate_csv(csv_dir="player_gamelogs", is_iso8601=True)
    league_gamelogs_instance = LeCommons.consolidate_csv(csv_dir="league_gamelogs", is_iso8601=False)

    player_gamelogs_df = player_gamelogs_instance.filter_by_player("austin reaves")
    league_gamelogs_df = league_gamelogs_instance.dataframe

    is_player_gamelog_healthy = player_gamelogs_instance.check_player_continuity(player_df=player_gamelogs_df)

    if not is_player_gamelog_healthy:
        raise ValueError(f"{debug_print} There is an issue with player continuity.")

    player_matchup_context = PlayerMatchUp(player_gamelogs_df_arg=player_gamelogs_df, league_gamelogs_df_arg=league_gamelogs_df)

    player_matchup_context.train_matchup_context_submodel(bet_line_under_or_over="over", points_or_equiv=13)
