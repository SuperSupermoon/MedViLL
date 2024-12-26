# MedViLL

Jong Hak Moon, Hyungyung Lee, Woncheol Shin, Young-Hak Kim, Edward Choi ([Paper](https://ieeexplore.ieee.org/abstract/document/9894658))


This repository provides the code for MedViLL(Medical Vision Language Learner).

---
<p align="center"><img src="https://user-images.githubusercontent.com/47732974/149651882-bb691bc8-8343-4699-a45f-1952bd558490.png")</p>
 
Our proposed architecture MedViLL is a single BERT-based model that learns unified contextualized vision-language (VL) representation for both Vision Language Understanding (VLU) and Vision Language Generation (VLG). MedViLL performs pre-training with a CNN-based visual encoder and a cross-modal Transformer for VL joint representation learning. After pre-training, our model can be easily used for VLU and VLG tasks with task-specific finetuning. Please refer to our paper "[**Multi-modal Understanding and Generation for Medical Images and Text via Vision-Language Pre-Training**](https://arxiv.org/abs/2105.11333)" for more details.

 
## 1) Downloads.
### Pre-trained weights.
We provide five versions of BERT-based pre-trained weights with different types of self-attention masks. Pre-training for the joint embedding was built on the BERT-base architecutre(12 hidden layers, 12 attention heads, 768 hidden size), and training details are described in our paper. Currently avaliable versions of pre-trained weights are as follows:
 
- [MedViLL](https://drive.google.com/file/d/1shOQrOWbkIeUUsQN48fEP6wj0e266jOb/view?usp=sharing) - BERT-Base model with Bidirectional Auto-regressive attention mask.

- [Bi & Seq2Seq](https://drive.google.com/file/d/1hn8DLgPkblIew_UEP3TwoLwKZw03Pkmk/view?usp=sharing) - BERT-Base model with Seq2Seq attention mask(75%) and Bidirectional attention mask(25%) in every mini-batch.

- [Bidirectional](https://drive.google.com/file/d/1GSb-CjUnfuTTDrb0tPEwHGo1Qg1JHvdf/view?usp=sharing) - BERT-Base model with Bidirectional attention mask.

- [Seq2Seq](https://drive.google.com/file/d/1O76qXkRkP-yS5iChwpH-8Z5EWDLbkWuu/view?usp=sharing) - BERT-Base model with Seq2Seq attention mask.

- [Non-cross](https://drive.google.com/file/d/1ZEu0NioO6ThJC_pWRYByyJ4-XwMnGvJA/view?usp=sharing) - BERT-Base model with Non-cross modality attention mask.


### Datasets.
We provide a pre-processed version of multiple datasets for each task as follows:

 Download each dataset to the path /data/preprocessed/[dataset].
- MIMIC-CXR: We don't provide MIMIC-CXR dataset due to the policy of data use agreement. Please download original datset from [PhysioNet](https://physionet.org/content/mimic-cxr-jpg/2.0.0/).
- [OPEN-I](https://drive.google.com/file/d/1aAKW2UcR7KhX9rckYtNfTfzNYulgrzle/view?usp=sharing) (74.1 MB): Unique study of 3,547 AP and PA image-report pairs from the official Open-I dataset.
- [VQA-RAD](https://drive.google.com/file/d/1zlNM7kQACaorfQD8n_Qtc5wkV_lh_60V/view?usp=sharing) (402 MB): 3,515 question answer pairs on 315 images (104 head CTs or MRIs, 107 Chest X-rays, and 104 abdominal CTs).
 
We also provide the JSON file with the path for validation in the retrieval task, download each files to the path /data/[dataset].
**Image to report retrieval**
1) [MIMIC valid](https://drive.google.com/file/d/1r9NMdZEDDjIi5L3EijTzKU13bluPEIIu/view?usp=sharing), 2) [MIMIC test](https://drive.google.com/file/d/1N4zaZrAYg6gjFR2yoEUFcwycjLNXc9FL/view?usp=sharing), 3) [OpenI test](https://drive.google.com/file/d/1GtKIlF9HSGzgA_yaVmoUsIs-ccOzonIz/view?usp=sharing)

**Report to Image retrieval**
1) [MIMIC valid](https://drive.google.com/file/d/1HBbq5Juxf_uh4Yk7SJTWoUH7yeyQfXGk/view?usp=sharing), 2) [MIMIC test](https://drive.google.com/file/d/11UQOId3-ErT-hkKSOKYQYUT7WrCYCywf/view?usp=sharing), 3) [OpenI test](https://drive.google.com/file/d/1CJkkDu4djlkeUTgZX7w3h1GC-IkPlgxh/view?usp=sharing)
 
 
## 2) Reproduce.
### Section A. Installation
Sections below describe the virtual env installation and the fine-training process of MedviLL based on pytorch version 1.7, python version 3.8. 
To fine-tune MedViLL, you need to download the pre-trained weights of MedViLL. After downloading the pre-trained weights, use medvill.yaml to install conda based virtual env as follows:

```
$ git clone https://github.com/SuperSupermoon/MedViLL.git
$ cd MedViLL; conda env create --file medvill.yaml
```

Note that all fine-tuning models were conducted on 8 Geforce RTX-3090 GPU machines, each of which has 24GB of VRAM. 

### Section B. Prepare pre-processed dataset

Unzip mimic, openi, and VQA-RAD tar.gz files. 

```
$ cd MedViLL; tar -zxvf [file_name.tar.gz]
```

### Section C. Pre-training model
Example:
```
$ cd MedViLL
$ python main.py
```


### Section D. Downstream model
- Diagnosis Classification
Example:
```
$ cd MedViLL/downstream_task/classification
$ python cls.py
```

- Image-Report Retrieval
Example:
```
$ cd MedViLL/downstream_task/retrieval
$ python retrieval.py
```

- Medical Visual Qestion Answering
Example:
```
$ python -m torch.distributed.launch --nproc_per_node=1 --master_port 9872 --use_env downstream_task/report_generation_and_vqa/finetune.py --model_recover_path pt_model/model.50.bin --tasks vqa --s2s_prob 0 --bi_prob 1 --mask_prob 0 --vqa_rad chest --vqa_eval
```

- Report Generation
Example:
```
$ cd MedViLL/downstream_task/report_generation_and_vqa
$ python finetune.py --tasks report_generation --mask_prob 0.15 --s2s_prob 1 --bi_prob 0
```
