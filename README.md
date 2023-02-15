# MUMR-Ubiquitous_Recommendation
These are the dataset and implementation for the paper:

*Jiayu Li, Zhiyu He, Yumeng Cui, Chenyang Wang, Chong Chen, Chun Yu, Min Zhang, Yiqun Liu, and Shaoping Ma, 2022. Towards Ubiquitous Personalized Music Recommendation with Smart Bracelets. In IMWUT2022*

Please cite the paper if you use the datasets or codes.

```
@article{li2022towards,
  title={Towards Ubiquitous Personalized Music Recommendation with Smart Bracelets},
  author={Li, Jiayu and He, Zhiyu and Cui, Yumeng and Wang, Chenyang and Chen, Chong and Yu, Chun and Zhang, Min and Liu, Yiqun and Ma, Shaoping},
  journal={Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies},
  volume={6},
  number={3},
  pages={1--34},
  year={2022},
  publisher={ACM New York, NY, USA}
}
```

If you have any problem about this work or the dataset, please raise the issues or contact with Min Zhang at z-m@tsinghua.edu.cn.

## Dataset
We select 1000 music tracks from the Million Song Dataset, and perform a one-week field study with 30 participants. During the field study, participants would listen to the recommended music, record their mood and preference ratings as feedback. Detailed explanations are shown in `data/README.md`.

## Implementation Codes
To run the codes, first run: `pip install -r requirements.txt`

Examples for running the codes are shown in `src/run.sh`. 

```
cd src/
bash run.sh
```


