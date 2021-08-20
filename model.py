from cassiopeia.core.common import CassiopeiaGhost
from tensorflow.keras.layers import (LSTM, Embedding, Dense, Dropout, Add, Reshape)
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.keras import Input
from tensorflow.keras.models import Model
import tensorflow as tf
import os
import numpy as np
import ast
import logging

START_ID = 0
logging.basicConfig(level=logging.INFO)


class LookUp:

    def __init__(self, cass, embedding_dim=200, embedding_dim_champions=50, embedding_dim_spells=20,
                 embedding_dim_runes=200, max_item_length=6):
        self.embedding_dim = embedding_dim
        self.embedding_dim_champions = embedding_dim_champions
        self.embedding_dim_spells = embedding_dim_spells
        self.embedding_dim_runes = embedding_dim_runes
        self.max_item_length = max_item_length
        self.cass = cass
        self.item_lookup = {}
        self.item_revers_lookup = {}
        self.spell_lookup = {}
        self.spell_revers_lookup = {}
        self.champion_lookup = {}
        self.champion_revers_lookup = {}
        self.rune_lookup = {}
        self.rune_revers_lookup = {}
        self.champion_id_to_name = {}
        self.item_id_to_name = {}
        self.setup()
        logging.info("lookups generated")

    def setup(self):
        self.item_lookup, self.item_revers_lookup = LookUp._make_lookup(self.cass.get_items(), offset=1)
        self.item_lookup[0] = 0
        self.spell_lookup, self.spell_revers_lookup = LookUp._make_lookup(self.cass.get_summoner_spells())
        self.champion_lookup, self.champion_revers_lookup = LookUp._make_lookup(self.cass.get_champions())
        self.rune_lookup, self.rune_revers_lookup = LookUp._make_lookup(self.cass.get_runes())
        _, self.champion_id_to_name = LookUp.__make_lookup(self.cass.get_champions())
        _, self.item_id_to_name = LookUp.__make_lookup(self.cass.get_items())

    def __repr__(self):
        return f"Lookup({vars(self)})"

    @staticmethod
    def __make_lookup(_items: [CassiopeiaGhost]):
        _name_to_id, _id_to_name = dict(), dict()

        for _item in _items:
            _name_to_id[_item.name] = _item.id
            _id_to_name[_item.id] = _item.name

        return _name_to_id, _id_to_name

    @staticmethod
    def _make_lookup(_items: [CassiopeiaGhost], offset=0):
        _name_to_id, _id_to_name = dict(), dict()
        _lookup, _revers_lookup = dict(), dict()

        for _item in _items:
            _name_to_id[_item.name] = _item.id
            _id_to_name[_item.id] = _item.name

        for _index, _item_id in enumerate(_name_to_id.values()):
            _index = _index + offset
            _lookup[_item_id] = _index
            _revers_lookup[_index] = _item_id
        return _lookup, _revers_lookup


class ItemModel:

    def __init__(self, _lookup):
        tf.config.experimental.enable_mlir_graph_optimization()
        self.lookup = _lookup
        self.model = self.make_model()
        logging.info("created model")

    def load_weights(self, path):
        self.model.load_weights(path)

    def make_model(self):
        item_size = len(self.lookup.item_lookup)
        champion_size = len(self.lookup.champion_lookup)
        runes_size = len(self.lookup.rune_lookup) + 1
        spells_size = len(self.lookup.spell_lookup)

        # print(f"spells: {spells_size}, items: {item_size}, champions: {champion_size}, runes: {runes_size}")

        spells = Input(shape=(2,), name="spells")
        spells_embedded = Embedding(spells_size, self.lookup.embedding_dim_spells, mask_zero=True)(spells)
        spells1 = Dropout(0.5)(spells_embedded)
        spells2 = Dense(256, activation='relu')(spells1)
        spells_reshaped = Reshape((512,))(spells2)
        spells3 = Dense(256, activation='relu')(spells_reshaped)

        runes = Input(shape=(6,), name="runes")
        runes_embedded = Embedding(runes_size, self.lookup.embedding_dim_runes, mask_zero=True)(runes)
        runes1 = Dropout(0.5)(runes_embedded)
        runes2 = Dense(256, activation='relu')(runes1)
        runes_reshaped = Reshape((1536,))(runes2)
        runes3 = Dense(256, activation='relu')(runes_reshaped)

        allay_champions = Input(shape=(1,), name="allay_champion")
        allay_champions_embedded = Embedding(champion_size, self.lookup.embedding_dim_champions, mask_zero=True)(
            allay_champions)
        allay_champions1 = Dropout(0.5)(allay_champions_embedded)
        allay_champions2 = Dense(256, activation='relu')(allay_champions1)
        allay_champions_reshaped = Reshape((256,))(allay_champions2)
        allay_champions3 = Dense(256, activation='relu')(allay_champions_reshaped)

        allay_decoder1 = Add()([allay_champions3, spells3, runes3])
        allay_decoder2 = Dense(256, activation='relu')(allay_decoder1)

        enemy_champions = Input(shape=(5,), name="enemy_champions")
        enemy_champions_embedded = Embedding(champion_size, self.lookup.embedding_dim_champions, mask_zero=True)(
            enemy_champions)
        enemy_champions1 = Dropout(0.5)(enemy_champions_embedded)
        enemy_champions2 = Dense(256, activation='relu')(enemy_champions1)
        enemy_champions_reshaped = Reshape((1280,))(enemy_champions2)
        enemy_champions3 = Dense(256, activation='relu')(enemy_champions_reshaped)

        enemy_and_allay_decoder1 = Add()([allay_decoder2, enemy_champions3])
        enemy_and_allay_decoder2 = Dense(256, activation='relu')(enemy_and_allay_decoder1)

        items = Input(shape=(self.lookup.max_item_length,), name="items")
        items_embedded = Embedding(item_size, self.lookup.embedding_dim, mask_zero=True)(items)  # mask_zero ???
        se2 = Dropout(0.5)(items_embedded)
        se3 = LSTM(256)(se2)

        decoder1 = Add()([enemy_and_allay_decoder2, se3])
        decoder2 = Dense(256, activation='relu')(decoder1)
        outputs = Dense(item_size, activation='softmax')(decoder2)
        caption_model = Model(inputs=[allay_champions, spells, runes, enemy_champions, items], outputs=outputs)
        caption_model.compile(loss='categorical_crossentropy', optimizer='adam')

        return caption_model

    def generate_item_build(self, data):
        c, s, r, e, _ = data
        items = [START_ID]
        for i in range(self.lookup.max_item_length):
            sequence = [self.lookup.item_lookup[x] for x in items]
            sequence = pad_sequences([sequence], maxlen=self.lookup.max_item_length)[0]
            # print(sequence)
            yhat = self.model.predict(
                [c.reshape((1, 1)), s.reshape((1, 2)), r.reshape((1, 6)), e.reshape((1, 5)), sequence.reshape((1, 6))],
                verbose=0)
            yhat = np.argmax(yhat)
            item_id = self.lookup.item_revers_lookup[yhat]
            # print(yhat, item_id)
            items.append(item_id)
        final = items[1:]
        return final


class Data:

    def __init__(self, _lookup):
        self.lookup = _lookup

    def _parse_line(self, d, s, i_in, i_out, r, e, c, max_item_length, predict):
        if len(d['enemy_champions']) == 5 and all([self.lookup.rune_lookup.get(x, False) for x in d['runes']]):
            if not predict:
                _items = d['items']
                _items.insert(0, START_ID)
            for ii in range(1, max_item_length):
                if not predict:
                    in_seq, out_seq = [self.lookup.item_lookup[x] for x in _items[:ii]], self.lookup.item_lookup[_items[ii]]
                    in_seq = pad_sequences([in_seq], maxlen=self.lookup.max_item_length)[0]
                    out_seq = to_categorical([out_seq], num_classes=len(self.lookup.item_lookup))[0]
                    i_in.append(np.array(in_seq))
                    i_out.append(np.array(out_seq))
                c.append(np.array(self.lookup.champion_lookup[d['champion_id']]))
                s.append(
                    np.array([self.lookup.spell_lookup[x] for x in [d['summoner_spell_1'], d['summoner_spell_2']]]))
                r.append(np.array([self.lookup.rune_lookup[x] for x in d['runes']]))
                e.append(np.array([self.lookup.champion_lookup[x] for x in d['enemy_champions']]))

    def prepare_data_from_file(self, path="data/data.txt", max_item_length=6, predict=False):
        with open(path, "r") as fp:
            lines = fp.readlines()
            s, i_in, i_out, r, e, c = [], [], [], [], [], []
        for line in lines:
            d = ast.literal_eval(line)
            self._parse_line(d, s, i_in, i_out, r, e, c, max_item_length, predict)

        return [np.array(c), np.array(s), np.array(r), np.array(e), np.array(i_in)], np.array(i_out)

    def prepare_data_from_list_of_dict(self, data, max_item_length=6, predict=False):
        s, i_in, i_out, r, e, c = [], [], [], [], [], []
        for value in data:
            self._parse_line(value, s, i_in, i_out, r, e, c, max_item_length, predict)
        return [np.array(c), np.array(s), np.array(r), np.array(e), np.array(i_in)], np.array(i_out)
