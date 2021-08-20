import cassiopeia as cass
import pickle
from cassiopeia.data import GameMode, Season, GameType, Queue
import os
from tqdm import tqdm

exclude_items = ["Dark Seal", "Steel Shoulderguards", "Spectral Sickle", "Spellthief's Edge", "Relic Shield", "Boots",
                 "Doran's Shield", "Doran's Ring", "Doran's Blade", "Cull", "Hailblade", "Emberknife", "Stealth Ward",
                 "Oracle Lens", "Control Ward", "Farsight Alteration"]

cass.set_riot_api_key("")
cass.set_default_region("EUW")


def make_item_lookup(_cass):
    _name_to_id = {}
    _id_to_name = {}
    items = _cass.get_items()
    for item in items:
        if not item.builds_into and item.name not in exclude_items:
            _name_to_id[item.name] = item.id
            _id_to_name[item.id] = item
    return _name_to_id, _id_to_name


def analyse_team(team, _id_to_name, _summoner_names):
    valid_summoners = []
    for participant in team.participants:

        if participant.summoner.name not in _summoner_names:
            _summoner_names.add(participant.summoner.name)

        if (items := analyse_participants_items(participant, _id_to_name)) is not None:
            valid_summoners.append({"champion_id": participant.champion.id, "items": items,
                                    "summoner_spell_1": participant.summoner_spell_d.id,
                                    "summoner_spell_2": participant.summoner_spell_f.id,
                                    "runes": [rune.id for rune in participant.runes]})
    return valid_summoners


def analyse_participants_items(participant, _id_to_name):
    if len(participant.stats.items) >= 6:
        items = [item.id for item in participant.stats.items if item and _id_to_name.get(item.id, None)]
        if len(items) >= 6:
            return items
    return None


def make_team_comb(team):
    champions_ids = []
    for participant in team.participants:
        champions_ids.append(participant.champion.id)
    return champions_ids


def get_match_history_from_summoner(_summoner, _id_to_name, _game_ids, _summoner_names):
    _data = list()
    try:
        for match in tqdm(_summoner.match_history, desc=f"Match History from: {_summoner.name}"):
            if match.id not in _game_ids:
                _game_ids.add(match.id)
                if match.mode == GameMode.classic and match.season == Season.season_9 and match.type == GameType.matched and (
                        match.queue == Queue.ranked_solo_fives or match.queue == Queue.ranked_flex_fives or match.queue ==
                        Queue.ranked_flex_threes):
                    blue_team = analyse_team(match.blue_team, _id_to_name, _summoner_names)
                    red_team = analyse_team(match.red_team, _id_to_name, _summoner_names)
                    if blue_team:
                        red_team_comb = make_team_comb(match.red_team)
                        for c in blue_team:
                            c["enemy_champions"] = red_team_comb
                    if red_team:
                        blue_team_comb = make_team_comb(match.blue_team)
                        for c in red_team:
                            c["enemy_champions"] = blue_team_comb

                    _data = _data + blue_team + red_team
    except Exception as e:
        print(str(e))
    return _data


if __name__ == "__main__":
    name_to_id, id_to_item = make_item_lookup(cass)

    print(len(name_to_id))

    with open("data/name_to_id.pk", "wb") as fp:
        pickle.dump(name_to_id, fp)

    #with open("data/id_to_name.pk", "wb") as fp:
    #    pickle.dump(id_to_name, fp)

    game_ids = set()
    summoner_names = {"A Do Sola Diatas"}

    if os.path.exists("data/game_ids.pk"):
        with open("data/game_ids.pk", "rb") as fp:
            game_ids = pickle.load(fp)

    if os.path.exists("data/summoner_names.pk"):
        with open("data/summoner_names.pk", "rb") as fp:
            summoner_names = pickle.load(fp)

    for name in summoner_names:
        new_summoner_names = set()
        summoner = cass.get_summoner(name=name, region="EUW")
        data = get_match_history_from_summoner(summoner, id_to_item, game_ids, new_summoner_names)

        with open("data/data.txt", "a") as fp:
            fp.writelines([str(x) + "\n" for x in data])

        with open("data/game_ids.pk", "wb") as fp:
            pickle.dump(game_ids, fp)

        with open("data/summoner_names.pk", "wb") as fp:
            new_summoner_names.update(summoner_names)
            pickle.dump(new_summoner_names, fp)
