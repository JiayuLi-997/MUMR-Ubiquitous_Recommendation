# Dataset

We select 1000 music tracks from the Million Song Dataset, and perform a one-week field study with 30 participants. During the field study, participants would listen to the recommended music, record their mood and preference ratings as feedback.
The music metadata is recorded in `music_metadata/`, the field study records are saved in `field_study/` after anonymization, and the dataset split settings are saved in `split/`.

## Music Metadata
**music_metadata/music_info.csv**

Original information of all music tracks. Each line contains the metadata of a track. There are 22 columns, including music id, music genre, and 20 features in Table 1. 

**music_metadata/music.json**

Json format of music information. Same as `music_info.csv`, where the music id is used as dict keys, and 20 features show in the dict values.

**music_metadata/music_norm.json**

Music metadata with normalized features. Used for model training and testing.

## Field Study Records
**field_study/interaction.json**

**field_study/label.json**

**field_study/user.json**

**field_study/env.json**

**field_study/env_norm.json**

**field_study/wrist.npy**

**field_study/wrist_norm.npy**

## Dataset Split
**split/CV/**

**split/LOSO**
