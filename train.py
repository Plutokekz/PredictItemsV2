import logging

from model import LookUp, Data, ItemModel
import os
import cassiopeia as cass
import time


def train(_lookup, _x, _y, batch_size=32, epochs=1, verbose=1, pretrained_weights_path="1629201863.5133631.hdf5"):
    item_model = ItemModel(_lookup)
    if os.path.exists(os.path.join("models", pretrained_weights_path)):
        item_model.model.load_weights(os.path.join("models", pretrained_weights_path))
    item_model.model.fit(x=_x, y=_y, batch_size=batch_size, epochs=epochs, verbose=verbose)
    item_model.model.save_weights(os.path.join("models", f"{str(time.time())}.hdf5"))
    return item_model


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    cass.set_default_region("EUW")

    lookup = LookUp(cass)

    data = Data(lookup)

    x, y = data.prepare_data_from_file()
    # x = [c, s, r, e, in_items]
    # y = out_items

    model = train(lookup, x, y)

    c, s, r, e, in_items = x

    for i in range(0, 60, 6):
        items = model.generate_item_build([c[i], s[i], r[i], e[i], in_items[i]])
        logging.info(f"Champion: {lookup.champion_id_to_name[lookup.champion_revers_lookup[c[i]]]}, Items: {[lookup.item_id_to_name[x] for x in items]}")
        logging.info(f"Enemy Team: {[lookup.champion_id_to_name[lookup.champion_revers_lookup[x]] for x in e[i]]}")
        print(f"Champion: {lookup.champion_id_to_name[lookup.champion_revers_lookup[c[i]]]}, Items: {[lookup.item_id_to_name[x] for x in items]}")
        print(f"Enemy Team: {[lookup.champion_id_to_name[lookup.champion_revers_lookup[x]] for x in e[i]]}")
