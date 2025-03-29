import polars
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
import numpy
import xgboost
from tqdm import tqdm
import multiprocessing


class LeTrainerUtils:
    def __init__(self):
        pass

    @staticmethod
    def train_xgboost(feature_extractions_arg: list[dict], n_iter: int) -> tuple[list[float | int], numpy.floating, numpy.floating]:

        def update_pbar(_study, _trial):
            progress_bar.update(1)

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # ===== PREPARE POLARS DATAFRAME. =====

        data_df = polars.DataFrame(feature_extractions_arg)
        sort_data_df = data_df.sort("game_date", descending=False)

        x = sort_data_df.drop(["game_date", "target"]).to_numpy()
        y = sort_data_df.get_column("target").to_numpy()

        # ===== CHECK PRECENTAGE SPLITS OF TARGET DATA. =====

        target_balance_check = (
            sort_data_df.get_column("target").value_counts().with_columns((polars.col("count") / polars.sum("count") * 100).alias("percentage"))
        )

        percentage_1 = target_balance_check.filter(polars.col("target") == 1).to_dict(as_series=False)
        percentage_0 = target_balance_check.filter(polars.col("target") == 0).to_dict(as_series=False)

        print(f"🐝 There are {percentage_1['count'][0]} with class {percentage_1['target'][0]} which is {percentage_1['percentage'][0]:.2f}")
        print(f"🐝 There are {percentage_0['count'][0]} with class {percentage_0['target'][0]} which is {percentage_0['percentage'][0]:.2f}")

        # ===== SET UP OUTER LOOP (TIME SERIES). =====

        outer_n_splits = 21
        outer_test_size = 1

        outer_tscv = TimeSeriesSplit(n_splits=outer_n_splits, test_size=outer_test_size)
        outer_scores = []
        # outer_predict = []
        # outer_predict_prob = []

        # ===== LOOP OVER OUTER FOLDS. =====

        for train_idx, val_idx in outer_tscv.split(x):

            progress_bar = tqdm(total=n_iter, desc="Optuna Trials")

            # ===== SPLIT TRAIN DATA. =====

            x_train_outer, y_train_outer = x[train_idx], y[train_idx]
            x_val_outer, y_val_outer = x[val_idx], y[val_idx]

            # ===== SET UP INNER LOOP (TIME SERIES). =====

            inner_n_splits = 21
            inner_test_size = 1

            inner_tscv = TimeSeriesSplit(n_splits=inner_n_splits, test_size=inner_test_size)

            n_jobs = int(multiprocessing.cpu_count() / 2)
            print(f"🐝 Number of cores to be used for hyperparameter sweep: {n_jobs}")

            # ===== DEFINE OPTUNA OBJECTIVE. =====

            def optuna_objective(trial: optuna.Trial) -> numpy.floating:
                optuna_scores = []

                param_sweep = {
                    "max_depth": trial.suggest_int("max_depth", 2, 5),
                    "learning_rate": trial.suggest_categorical("learning_rate", [0.01, 0.02, 0.05, 0.1]),
                    "n_estimators": trial.suggest_categorical("n_estimators", [50, 100, 200, 300, 500]),
                    "subsample": trial.suggest_categorical("subsample", [0.6, 0.8, 1.0]),
                    "colsample_bytree": trial.suggest_categorical("colsample_bytree", [0.6, 0.8, 1.0]),
                    "colsample_bylevel": trial.suggest_categorical("colsample_bylevel", [0.6, 1.0]),
                    "colsample_bynode": trial.suggest_categorical("colsample_bynode", [0.6, 1.0]),
                    "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                    "gamma": trial.suggest_categorical("gamma", [0, 0.1, 0.5, 1]),
                    "reg_alpha": trial.suggest_categorical("reg_alpha", [0, 0.001, 0.01, 0.1, 1, 10]),
                    "reg_lambda": trial.suggest_categorical("reg_lambda", [0.1, 1.0, 2.0, 5.0, 10]),
                    "scale_pos_weight": trial.suggest_categorical("scale_pos_weight", [1, 2, 3, 5]),
                    # "booster": trial.suggest_categorical("booster", ["gbtree", "dart"]),
                }

                for inner_train_idx, inner_val_idx in inner_tscv.split(x_train_outer):
                    x_train_inner, y_train_inner = x_train_outer[inner_train_idx], y_train_outer[inner_train_idx]
                    x_val_inner, y_val_inner = x_train_outer[inner_val_idx], y_train_outer[inner_val_idx]

                    optuna_objective_model = xgboost.XGBClassifier(
                        objective="binary:logistic",
                        eval_metric="logloss",
                        verbosity=0,
                        # ===== CUDA support. =====
                        device="cuda",
                        tree_method="hist",
                        # sampling_method=sampling_method,
                        # ===== Param sweep. =====
                        **param_sweep,
                    )

                    optuna_objective_model.fit(x_train_inner, y_train_inner)

                    # ===== Run predictions and evaluate. =====

                    y_predict_inner = optuna_objective_model.predict(x_val_inner)
                    accuracy_score_optuna = accuracy_score(y_val_inner, y_predict_inner)

                    optuna_scores.append(accuracy_score_optuna)

                return numpy.mean(optuna_scores)

            # ===== RUN OPTUNA. =====

            optuna_sampler = optuna.samplers.TPESampler()
            pruner = optuna.pruners.MedianPruner(n_startup_trials=10)

            study = optuna.create_study(sampler=optuna_sampler, pruner=pruner, direction="maximize")
            study.optimize(optuna_objective, n_jobs=n_jobs, n_trials=n_iter, show_progress_bar=False, callbacks=[update_pbar])

            progress_bar.close()

            optuna_best_params = study.best_params
            optuna_best_accuracy_found = study.best_value

            print(f"🐝 Optuna best params accuracy: {optuna_best_accuracy_found:.4f}")
            print(f"🐝 Optuna best params: {optuna_best_params}")

            # ===== RETRAIN ON OUTERLOOP WITH BEST PARAMS FROM OPTUNA. =====

            best_param_model = xgboost.XGBClassifier(**optuna_best_params, objective="binary:logistic", eval_metric="logloss", verbosity=0)
            best_param_model.fit(x_train_outer, y_train_outer)

            # ===== PREDICT AND EVALUATE. =====

            y_pred = best_param_model.predict(x_val_outer)
            # y_pred_prob = best_param_model.predict_proba(x_val_outer)[:, 1]

            best_param_accuracy = accuracy_score(y_true=y_val_outer, y_pred=y_pred)

            outer_scores.append(best_param_accuracy)
            # outer_predict.append(y_pred)
            # outer_predict_prob.append(y_pred_prob)

        # ===== OVERALL PERFORMANCE (AVERAGE ACROSS OUTER FOLDS). =====

        # print(f"\n🍯 ===== FINAL PERFORMANCE ===== 🍯\n")
        # print(f"Outer fold accuracy scores:", outer_scores)
        # print(f"Mean across outer folds acuracy:", numpy.mean(outer_scores))
        # print(f"Standard deviation mean across outer folds:", numpy.std(outer_scores))

        # results_converted = []
        #
        # for true_class, pred, prob in zip(y, outer_predict, outer_predict_prob):
        #     confidence = abs(prob - 0.5) * 2
        #
        #     results_converted.append(
        #         {
        #             "true_class": true_class,
        #             "predicted_class": pred,
        #             "prob_class_1": prob,
        #             "confidence": confidence,
        #             "correct": pred == true_class,
        #         }
        #     )
        #
        # results_converted_df = polars.DataFrame(results_converted)
        #
        # # ----- Evaluate high confidence predictions. -----
        #
        # confidence_threshold = 0.70
        #
        # high_confidence_df = results_converted_df.filter(polars.col("confidence") >= confidence_threshold)
        #
        # if high_confidence_df.height == 0:
        #     print(f"🐝 There were no samples that meet the confidence threshold of {confidence_threshold}")
        #     return
        #
        # accuracy_high_confidence = high_confidence_df.get_column("correct").mean()
        #
        # print(f"🐝 Accuracy of high-confidence predictions: {accuracy_high_confidence}")
        # print(high_confidence_df)

        return outer_scores, numpy.mean(outer_scores), numpy.std(outer_scores)
