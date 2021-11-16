## Model Zoo and Results

### Environment
- 4/1 NVIDIA V100 GPUs for training/evaluation.
- auto-mixed precision was enabled in training but disabled in evaluation.
- test-time augmentations were not used.

### Pre-trained Models
`PRE`: the pre-training stage with static images.
`PRE_YTB_DAV`: the main-training stage with YouTube-VOS and DAVIS, used for evaluation.
| Model      | Param (M) |                                             PRE                                              |                                         PRE_YTB_DAV                                          |
|:---------- |:---------:|:--------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------:|
| AOTT       |    5.7    | [gdrive](https://drive.google.com/file/d/1_513h8Hok9ySQPMs_dHgX5sPexUhyCmy/view?usp=sharing) | [gdrive](https://drive.google.com/file/d/1owPmwV4owd_ll6GuilzklqTyAd0ZvbCu/view?usp=sharing) |
| AOTS       |    7.0    | [gdrive](https://drive.google.com/file/d/1QUP0-VED-lOF1oX_ppYWnXyBjvUzJJB7/view?usp=sharing) | [gdrive](https://drive.google.com/file/d/1beU5E6Mdnr_pPrgjWvdWurKAIwJSz1xf/view?usp=sharing) |
| AOTB       |    8.3    | [gdrive](https://drive.google.com/file/d/11Bx8n_INAha1IdpHjueGpf7BrKmCJDvK/view?usp=sharing) | [gdrive](https://drive.google.com/file/d/1hH-GOn4GAxHkV8ARcQzsUy8Ax6ndot-A/view?usp=sharing) |
| AOTL       |    8.3    | [gdrive](https://drive.google.com/file/d/1WL6QCsYeT7Bt-Gain9ZIrNNXpR2Hgh29/view?usp=sharing) | [gdrive](https://drive.google.com/file/d/1L1N2hkSPqrwGgnW9GyFHuG59_EYYfTG4/view?usp=sharing) |
| R50-AOTL   |   14.9    | [gdrive](https://drive.google.com/file/d/1hS4JIvOXeqvbs-CokwV6PwZV-EvzE6x8/view?usp=sharing) | [gdrive](https://drive.google.com/file/d/1qJDYn3Ibpquu4ffYoQmVjg1YCbr2JQep/view?usp=sharing) |
| SwinB-AOTL |   65.4    | [gdrive](https://drive.google.com/file/d/1LlhKQiXD8JyZGGs3hZiNzcaCLqyvL9tj/view?usp=sharing) | [gdrive](https://drive.google.com/file/d/192jCGQZdnuTsvX-CVra-KVZl2q1ZR0vW/view?usp=sharing) |

### YouTube-VOS 2018 val
`ALL-F`: all frames. The default evaluation setting of YouTube-VOS is 6fps, but 30fps sequences are also supplied. We noticed that many VOS methods prefer to evaluate with 30fps videos. Thus, we also supply our results here.
| Model      |    Stage    |   FPS    | All-F |   Mean   |  J Seen  | J Unseen |  F seen  | F unseen | Prediction |
|:---------- |:-----------:|:--------:|:-----:|:--------:|:--------:|:--------:|:--------:|:--------:|:----:|
| AOTT       | PRE_YTB_DAV | **41.0** |   √   |   80.9   |   80.0   |   84.7   |   75.2   |   83.5   | Coming |
| AOTT       | PRE_YTB_DAV | **41.0** |       |   80.2   |   80.4   |   85.0   |   73.6   |   81.7   | Coming |
| AOTS       | PRE_YTB_DAV |   27.1   |   √   |   83.0   |   82.2   |   87.0   |   77.3   |   85.7   | Coming |
| AOTS       | PRE_YTB_DAV |   27.1   |       |   82.9   |   82.3   |   87.0   |   77.1   |   85.1   | Coming |
| AOTB       | PRE_YTB_DAV |   20.5   |   √   |   84.1   |   83.6   |   88.5   |   78.0   |   86.5   | Coming |
| AOTB       | PRE_YTB_DAV |   20.5   |       |   84.0   |   83.2   |   88.1   |    78    |   86.5   | Coming |
| AOTL       | PRE_YTB_DAV |   6.5    |   √   |   84.5   |   83.7   |   88.8   |   78.4   |   87.1   | Coming |
| AOTL       | PRE_YTB_DAV |   16.0   |       |   84.1   |   83.2   |   88.2   |   78.2   |   86.8   | Coming |
| R50-AOTL   | PRE_YTB_DAV |   6.4    |   √   | **85.5** |   84.5   |   89.5   | **79.6** | **88.2** | Coming |
| R50-AOTL   | PRE_YTB_DAV |   15.1   |       |   84.6   |   83.7   |   88.5   |   78.8   |   87.3   | Coming |
| SwinB-AOTL | PRE_YTB_DAV |   5.2    |   √   |   85.1   | **85.1** | **90.1** |   78.4   |   86.9   | Coming |
| SwinB-AOTL | PRE_YTB_DAV |   12.1   |       |   84.7   |   84.5   |   89.5   |   78.1   |   86.7   | Coming |

### YouTube-VOS 2019 val
| Model      |    Stage    |   FPS    | All-F |   Mean   |  J Seen  | J Unseen |  F seen  | F unseen | Prediction |
|:---------- |:-----------:|:--------:|:-----:|:--------:|:--------:|:--------:|:--------:|:--------:|:----:|
| AOTT       | PRE_YTB_DAV | **41.0** |   √   |   80.9   |   79.9   |   84.4   |   75.6   |   83.8   | Coming |
| AOTT       | PRE_YTB_DAV | **41.0** |       |   80.0   |   79.8   |   84.2   |   74.1   |   82.1   | Coming |
| AOTS       | PRE_YTB_DAV |   27.1   |   √   |   82.8   |   81.9   |   86.5   |   77.3   |   85.6   | Coming |
| AOTS       | PRE_YTB_DAV |   27.1   |       |   82.7   |   81.9   |   86.5   |   77.3   |   85.2   | Coming |
| AOTB       | PRE_YTB_DAV |   20.5   |   √   |   84.1   |   83.3   |   88.0   |   78.2   |   86.7   | Coming |
| AOTB       | PRE_YTB_DAV |   20.5   |       |   84.0   |   83.1   |   87.7   |   78.5   |   86.8   | Coming |
| AOTL       | PRE_YTB_DAV |   6.5    |   √   |   84.2   |   83.0   |   87.8   |   78.7   |   87.3   | Coming |
| AOTL       | PRE_YTB_DAV |   16.0   |       |   84.0   |   82.8   |   87.6   |   78.6   |   87.1   | Coming |
| R50-AOTL   | PRE_YTB_DAV |   6.4    |   √   | **85.3** |   83.9   |   88.8   | **79.9** | **88.5** | Coming |
| R50-AOTL   | PRE_YTB_DAV |   15.1   |       |   84.4   |   83.4   |   88.1   |   78.7   |   87.2   | Coming |
| SwinB-AOTL | PRE_YTB_DAV |   5.2    |   √   | **85.3** | **84.6** | **89.5** |   79.3   |   87.7   | Coming |
| SwinB-AOTL | PRE_YTB_DAV |   12.1   |       |   84.7   |   84.0   |   88.8   |   78.7   |   87.1   | Coming |

### DAVIS-2017 test

| Model      |    Stage    | FPS  |   Mean   | J score  | F score  | Prediction |
| ---------- |:-----------:|:----:|:--------:|:--------:|:--------:|:----:|
| AOTT       | PRE_YTB_DAV | **51.4** |   73.7   |   70.0   |   77.3   | Coming |
| AOTS       | PRE_YTB_DAV | 40.0 |   75.2   |   71.4   |   78.9   | Coming |
| AOTB       | PRE_YTB_DAV | 29.6 |   77.4   |   73.7   |   81.1   | Coming |
| AOTL       | PRE_YTB_DAV | 18.7 |   79.3   |   75.5   |   83.2   | Coming |
| R50-AOTL   | PRE_YTB_DAV | 18.0 |   79.5   |   76.0   |   83.0   | Coming |
| SwinB-AOTL | PRE_YTB_DAV | 9.3  | **82.1** | **78.2** | **85.9** | Coming |

### DAVIS-2017 val

| Model      |    Stage    | FPS  |   Mean   | J score  |  F score  | Prediction |
| ---------- |:-----------:|:----:|:--------:|:--------:|:---------:|:----:|
| AOTT       | PRE_YTB_DAV | **51.4** |   79.2   |   76.5   |   81.9    | Coming |
| AOTS       | PRE_YTB_DAV | 40.0 |   82.1   |   79.3   |   84.8    | Coming |
| AOTB       | PRE_YTB_DAV | 29.6 |   83.3   |   80.6   |   85.9    | Coming |
| AOTL       | PRE_YTB_DAV | 18.7 |   83.6   |   80.8   |   86.3    | Coming |
| R50-AOTL   | PRE_YTB_DAV | 18.0 |   85.2   |   82.5   |   87.9    | Coming |
| SwinB-AOTL | PRE_YTB_DAV | 9.3  | **85.9** | **82.9** | **88.9** | Coming |

### DAVIS-2016 val

| Model      |    Stage    | FPS  |   Mean   | J score  | F score  | Prediction |
| ---------- |:-----------:|:----:|:--------:|:--------:|:--------:|:----:|
| AOTT       | PRE_YTB_DAV | **51.4** |   87.5   |   86.5   |   88.4   | Coming |
| AOTS       | PRE_YTB_DAV | 40.0 |   89.6   |   88.6   |   90.5   | Coming |
| AOTB       | PRE_YTB_DAV | 29.6 |   90.9   |   89.6   |   92.1   | Coming |
| AOTL       | PRE_YTB_DAV | 18.7 |   91.1   |   89.5   |   92.7   | Coming |
| R50-AOTL   | PRE_YTB_DAV | 18.0 |   91.7   |   90.4   |   93.0   | Coming |
| SwinB-AOTL | PRE_YTB_DAV | 9.3  | **92.2** | **90.6** | **93.8** | Coming |
