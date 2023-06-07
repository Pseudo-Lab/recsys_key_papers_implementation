SONG_DATASET_DIR = 'song_dataset/'
SONG_DATA_DIR = 'song_data/'
SONG_IX_DATA_DIR = 'song_data_ix/'
SONG_IX_MAPPING_DIR = 'song_ix_mapping/'
PATH_DATA_DIR = 'path_data/'

PERSON_SONG_DICT = 'person_song.dict'
SONG_PERSON_DICT = 'song_person.dict'
USER_SONG_DICT = 'user_song.dict'
SONG_USER_DICT = 'song_user.dict'

TYPE_TO_IX = 'type_to_ix.dict'
RELATION_TO_IX = 'relation_to_ix.dict'
ENTITY_TO_IX = 'entity_to_ix.dict'
IX_TO_TYPE = 'ix_to_type.dict'
IX_TO_RELATION = 'ix_to_relation.dict'
IX_TO_ENTITY = 'ix_to_entity.dict'

PAD_TOKEN = '#PAD_TOKEN'
SONG_TYPE = 0
USER_TYPE = 1
PERSON_TYPE = 2
PAD_TYPE = 3

SONG_PERSON_REL = 0
PERSON_SONG_REL = 1
USER_SONG_REL = 2
SONG_USER_REL = 3
UNK_REL = 4
END_REL = 5
PAD_REL = 6

ENTITY_EMB_DIM = 64 #64 in paper
TYPE_EMB_DIM =32 #32 in paper
REL_EMB_DIM = 32 #32 in paper
HIDDEN_DIM = 256 #256 in paper
TAG_SIZE = 2 #since 0 or 1

MAX_PATH_LEN = 6
NEG_SAMPLES_TRAIN = 4
NEG_SAMPLES_TEST = 100

LEN_3_BRANCH = 50 #branching factor for paths
LEN_5_BRANCH_TRAIN = 6
LEN_5_BRANCH_TEST= 10
