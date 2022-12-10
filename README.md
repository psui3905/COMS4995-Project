# COMS4995-Project 

[![Next][Pytorch]][Next-url]
[![Next][gcloud]][gcloud-url]

A Hybrid Ensembled Solution for SIIM-FISABIO-RSNA COVID-19 Detection 


## Environment setup
- HARDWARE: 1 x NVIDIA GeForce 3090 24G GPU, 32 vCPUs, 78GB RAM
- Ubuntu 18.04.6 LTS
- CUDA 11.4
- Python 3.9.12

```
$ sudo apt-get --assume-yes update
$ sudo apt-get --assume-yes install git man wget curl unzip zip fish htop tmux
$ sudo apt-get update -y && apt-get install software-properties-common -y && \
    add-apt-repository -y multiverse && apt-get update -y && apt-get upgrade -y && \
    apt-get install -y apt-utils nano vim man build-essential wget sudo && \
    rm -rf /var/lib/apt/lists/*
$ sudo apt-get --assume-yes install ffmpeg libsm6 libxext6  -y
$ sudo apt-get update
```

## Dataset
- Download processed SIIM dataset from google bucket:
```
$ echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
$ apt-get install apt-transport-https ca-certificates gnupg 
$ curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
$ apt-get update && sudo apt-get install google-cloud-sdk 
$ apt-get install google-cloud-sdk-app-engine-python --assume-yes 
$ gcloud init 
$ gsutil cp gs://siimcovid19_jpg/siim_512_jpg.zip .
$ unzip siim_512_jpg.zip
```


## Getting Started

```
$ git clone https://github.com/psui3905/COMS4995-Project.git
$ mv jpg_form ~/COMS4995-Project
$ cd COMS4995-Project
$ pip install -r requirement.txt
$ python3 train.py
```

## Contribution
This project is equally contributed by Pengwei Sui (ps3307) and Zheyu Zhang (zz2980).

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/psui3905/COMS4995-Project.svg?style=for-the-badge
[contributors-url]: https://github.com/psui3905/COMS4995-Project/graphs/contributors

[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: images/screenshot.png

[Pytorch]: https://img.shields.io/badge/Pytorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white
[Pytorch-url]: https://pytorch.org/

[gcloud]: https://img.shields.io/badge/GoogleCloud-4285F4?style=for-the-badge&logo=googlecloud&logoColor=white
[gcloud-url]: https://pytorch.org/

[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 
