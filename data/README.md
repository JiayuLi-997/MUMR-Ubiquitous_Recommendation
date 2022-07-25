# Dataset

We select 1000 music tracks from the Million Song Dataset, and perform a one-week field study with 30 participants. During the field study, participants would listen to the recommended music, record their mood and preference ratings as feedback.
The music metadata is recorded in `music_metadata/`, the field study records are saved in `field_study/` after anonymization, and the dataset split settings are saved in `split/`.

## Music Metadata
**music_metadata/music_info.csv**

Original information of all music tracks. Each line contains the metadata of a track. There are 22 columns, including music id, music genre, and 20 features in Table 1. 

**music_metadata/music.json**

Json format of music information. Same as `music_info.csv`, where the music id is used as dict keys, and 20 features show in the dict values.

**music_metadata/music_norm.json**

Music metadata with normalized (for continous features) and one-hot embedding (for discrete features) features. Used for model training and testing.

## Field Study Records
**field_study/interaction.json**

Interaction records between users and music tracks. Each item represents an interaction between a user and a track.

```
Formatting:
{record id: [user id, music id]}
```

**field_study/label.json**

Rating and mood labels for each interaction. Each item contains mood and rating for an interaction.

Ratings are in the range of {1..5}, and moods are in the range of [-1,1].

```
Formatting:
{record id: [rating, mood_pre(valence), mood_pre(arousal), mood_post(valence), mood_post(arousal) ]}
```

**field_study/user.json**

One-hot embedding of user id.

```
Formatting:
{user id: one-hot embedding.}
```

**field_study/env.json**

Each item contains the environment contexts for an interaction. Please refer to *Section4.4 Data Pre-processing* in the paper for processing details about each feature.

```
Formatting:
{interaction id: {'time': time period, 'weather': [weather condition, pressure, temperature, humidity], 'GPS': [relative longitude,relative latitude, relative speed]}}
```

**field_study/env_norm.json**

Environment contexts with normalized (for continous features) and one-hot embedding (for discrete features) features. Used for model training and testing.

**field_study/wrist.npy**

Contexts collected with bracelets. Each line contains the bracelet contexts for an interaction, sorted by interaction id.

```
Formatting:
[relative heart rate, activity intensity, activity step, activity type]
```

**field_study/wrist_norm.npy**

Bracelet contexts with normalized (for continous features) and one-hot embedding (for discrete features) features. Used for model training and testing.


## Dataset Split
**split/CV/**

Dataset split for ten-fold cross validation. Each file contains an array of interactions ids for training or testing.

**split/LOSO**

Dataset split for Leave-one-subject-out cross validation.
