from pprint import pprint

import cassiopeia as cass
from cassiopeia import ChampionMastery
from datapipelines import NotFoundError
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.styles import Style

from model import LookUp, Data, ItemModel
from prompt_toolkit.shortcuts import input_dialog
from prompt_toolkit import Application, PromptSession
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.layout.containers import VSplit, Window, HSplit
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.key_binding import KeyBindings
import pickle

cass.apply_settings("config.json")
lookup = LookUp(cass)


# name = input_dialog(title='Input Summoner Name', text='name:').run()
class Api:

    def __init__(self, cass, lookup):
        self.cass: cass = cass
        self.lookup: LookUp = lookup

    def get_current_game(self, name="Dvek"):
        summoner = self.cass.get_summoner(name=name)

        try:
            current_match = summoner.current_match
            return summoner, current_match
        except NotFoundError:
            return summoner, None

    @classmethod
    def make_team_comb(cls, team):
        return [participant.champion.id for participant in team.participants]

    def get_champion(self, participant):
        return {"champion_id": participant.champion.id, "items": None,
                "summoner_spell_1": participant.summoner_spell_d.id,
                "summoner_spell_2": participant.summoner_spell_f.id,
                # TODO: runen sind mehr als 6 model nimmt aber nur 6 ???
                "runes": [rune.id for rune in participant.runes if self.lookup.rune_lookup.get(rune.id, False)][:6]}

    @classmethod
    def find_summoner(cls, team, summoner_name):
        for p in team.participants:
            if p.summoner.name == summoner_name:
                return p
        return False

    def get_data(self, match, summoner_name):
        if participant := self.find_summoner(match.blue_team, summoner_name):
            data = self.get_champion(participant)
            data["enemy_champions"] = self.make_team_comb(match.red_team)
            return data
        if participant := self.find_summoner(match.red_team, summoner_name):
            data = self.get_champion(participant)
            data["enemy_champions"] = self.make_team_comb(match.blue_team)
            return data


kb = KeyBindings()


@kb.add('c-q')
def exit_(event):
    """
    Pressing Ctrl-Q will exit the user interface.

    Setting a return value means: quit the event loop that drives the user
    interface and return this value from the `Application.run()` call.
    """
    event.app.exit()


class Handler:

    def __init__(self, api, item_text, summoner_text, model, file, data):
        self.api = api
        self.text: FormattedTextControl = item_text
        self.summoner_text: FormattedTextControl = summoner_text
        self.file = file
        self.model = model
        self.data: Data = data

    def format_summoner_text(self, summoner):
        ranks = "\n".join([f"    {str(q.value)}: {r.tier.value} {r.division.value}" for q, r in summoner.ranks.items()])
        m = summoner.champion_masteries
        m.sort(key=lambda item: item.points, reverse=True)
        masteries = "\n".join(
            [f"    {x.champion.name}, {x.level}, {x.points}" for x in summoner.champion_masteries[:5]])
        return FormattedText([('#ff0066 bold', 'Summoner:'), ('#44ff00', f'\n    {summoner.name}\n'),
                              ('#ff0066 bold', 'Level:'), ('#44ff00', f'\n    {summoner.level}\n'),
                              ('#ff0066 bold', 'rank:'), ('#44ff00', f'\n{ranks}\n'),
                              ('#ff0066 bold', 'champion masteries:'), ('#44ff00 italic blink', f'\n{masteries}\n')])

    def format_items_windows(self, items, champion, enemies):
        return FormattedText([('#ff0066 bold', 'Champion:'), ('#44ff00', f' {champion}\n'),
                              ('#ff0066 bold', 'Items:'),
                              ('#44ff00', f' {", ".join(items[:4])}\n       {", ".join(items[4:])}\n'),
                              ('#ff0066 bold', 'Enemies:'), ('#44ff00', f'\n{", ".join(enemies)}\n')])

    def username_input_handler(self, buffer: Buffer):
        summoner, current_match = self.api.get_current_game(buffer.text)
        if summoner:
            self.summoner_text.text = self.format_summoner_text(summoner)
            if summoner.name not in self.file:
                self.file.append(summoner.name)

        if summoner and current_match:
            data = self.api.get_data(current_match, summoner.name)

            self.text.text = str(data)

            data, _ = self.data.prepare_data_from_list_of_dict([data], 2, True)

            self.text.text += "\n" + str(data)

            items = self.model.generate_item_build(data)

            c, s, r, e, _ = data

            # print(
            #    f"Champion: {lookup.champion_id_to_name[lookup.champion_revers_lookup[c]]}, Items: {[lookup.item_id_to_name[x] for x in items]}")
            # print(f"Enemy Team: {[lookup.champion_id_to_name[lookup.champion_revers_lookup[x]] for x in e]}")

            # text = f"Champion: {lookup.champion_id_to_name[lookup.champion_revers_lookup[c[0]]]}, Items: {[lookup.item_id_to_name[x] for x in items]}"
            self.text.text = self.format_items_windows([lookup.item_id_to_name[x] for x in items],
                                                       lookup.champion_id_to_name[lookup.champion_revers_lookup[c[0]]],
                                                       [lookup.champion_id_to_name[lookup.champion_revers_lookup[x]] for
                                                        x in e[0]])
        elif summoner:
            self.text.text = f"Username {buffer.text} not found"
        return False


def main():
    api = Api(cass, lookup)
    model = ItemModel(lookup)
    data = Data(lookup)

    with open("auto_complete.pkl", "rb") as fp:
        file = pickle.load(fp)

    my_style = Style.from_dict({
        # User input (default text).
        '': '#ff0066',

        # Prompt.
        'username': '#884444',
        'at': '#00aa00',
        'colon': '#0000aa',
        'pound': '#00aa00',
        'host': '#00ffff bg:#444400',
        'path': 'ansicyan underline',
    })

    summoner_text = FormattedTextControl()
    item_text = FormattedTextControl()
    handler = Handler(api, item_text, summoner_text, model, file, data)
    summoner_name_completer = WordCompleter(file)
    buffer1 = Buffer(accept_handler=handler.username_input_handler, multiline=False, completer=summoner_name_completer,
                     complete_while_typing=True)  # Editable buffer.

    root_container = VSplit([
        # One window that holds the BufferControl with the default buffer on
        # the left.
        Window(content=BufferControl(buffer=buffer1)),

        # A vertical line in the middle. We explicitly specify the width, to
        # make sure that the layout engine will not try to divide the whole
        # width by three for all these windows. The window will simply fill its
        # content by repeating this character.
        Window(width=1, char='│'),

        HSplit([
            Window(content=summoner_text),
            Window(height=1, char='─'),
            Window(content=item_text),
        ]),

    ])

    layout = Layout(root_container)

    app = Application(key_bindings=kb, layout=layout, full_screen=True)
    app.run()  # You won't be able to Exit this app

    with open("auto_complete.pkl", "wb") as fp:
        pickle.dump(file, fp)


if __name__ == '__main__':
    main()
