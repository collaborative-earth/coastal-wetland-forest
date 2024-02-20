from src.data import GoogleStorageReader, Tile

def test_tile():
    bucket_name = 'dummy'
    data_bands=['SR_B2','SR_B3','SR_B4','SR_B5']
    label_name='cwf'
    tile = Tile(bucket_name,data_bands,label_name)
    assert tile.label_name == label_name
    assert tile.data_bands == data_bands
    assert tile.data_label_bands() == data_bands + [label_name]