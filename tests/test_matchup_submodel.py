import datetime
import pytest
from matchup_submodel import PlayerMatchUp
from pathlib import Path


@pytest.fixture()
def data_fixture():
    curr_dir = Path.cwd()
    league_gamelog = curr_dir.parent.joinpath("league_gamelog_validation_v2.csv")

    return {"league_gamelog": league_gamelog}


def test_train_jokic(data_fixture):
    curr_dir = Path.cwd()

    player_last = "jokic"
    jokic_gamelog = curr_dir.joinpath("validation", "jokic_v2.csv")
    datetime_check = datetime.datetime(2024, 11, 2, 0, 0, 0)

    matchup_instance = PlayerMatchUp(
        player_last_arg=player_last, player_gamelogs_csv_arg=jokic_gamelog, league_gamelogs_csv_arg=data_fixture["league_gamelog"]
    )
    matchup_instance.train_matchup_context_submodel()

    get_feat_extract = matchup_instance.feature_extraction

    assert len(get_feat_extract) == 1
    assert get_feat_extract[0]["game_date"] == datetime_check
    assert get_feat_extract[0]["opp_def_rating"] == pytest.approx(98.31531943)
    assert get_feat_extract[0]["opp_avg_pace"] == pytest.approx(100.032)
    assert get_feat_extract[0]["avg_opp_pts_allowed"] == pytest.approx(116.4)


def test_train_hayes(data_fixture):
    curr_dir = Path.cwd()

    player_last = "hayes"
    hayes_gamelog = curr_dir.joinpath("validation", "hayes_v1.csv")
    datetime_check = datetime.datetime(2024, 11, 1, 0, 0, 0)

    matchup_instance = PlayerMatchUp(
        player_last_arg=player_last, player_gamelogs_csv_arg=hayes_gamelog, league_gamelogs_csv_arg=data_fixture["league_gamelog"]
    )
    matchup_instance.train_matchup_context_submodel()

    get_hayes_feat_extract = matchup_instance.feature_extraction

    assert len(get_hayes_feat_extract) == 1
    assert get_hayes_feat_extract[0]["game_date"] == datetime_check
    assert get_hayes_feat_extract[0]["opp_def_rating"] == pytest.approx(113.3180784)
    assert get_hayes_feat_extract[0]["opp_avg_pace"] == pytest.approx(101.52576)
