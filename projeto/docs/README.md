![DeepBrain Logo](/projeto/docs/blendBoard.png)

# 1. Introdução
### 1.1 Panorama do projeto
O **Deep Brain** é uma aplicação de machine learning capaz de detectar tumores (glioma, meningioma e pituitário) em imagens de ressonância magnética do cérebro. 
A aplicação é baseada em uma rede neural convulacional treinada em imagens destes tipos de tumores. Esta rede retorna probabilidades associadas a cada tipo de tumor - ou à ausência de tumor - em uma imagem de ressonância fornecida pelo usuário.

### 1.2. Sobre tumores cerebrais
#### **Visão geral**
Um tumor cerebral consiste em um crescimento anômalo de células no cérebro. Ele pode se desenvolver tanto em tecidos cerebrais - como o glioma - ou em locais próximos, como a glândula 
pituitária ou nervos. Muitos tumores cerebrais não são malignos; no entanto, tratando-se de tumores malignos ou não, os tumores podem trazer uma série de problemas ao paciente, uma vez
que o crânio é uma estrutura rígida e, por isso, o espaço é bastante limitado. Especificamente, o crescimento de um tumor benigno, por exemplo, pode causar danos ao:
- Invadir tecido cerebral, destruindo-o;
- Pressionando tecidos adjacentes;
- Causando hemorragias;
- Bloqueando a circulação do fluido cerebroespinhal; entre outros.

#### **Sintomas**
Tanto tumores malignos quanto benignos causam sintomas, com a diferença de que os tumores benignos tendem a ter uma progressão lenta dos sintomas, enquanto os tumores malignos
evoluem rapidamente (em questão de dias ou semanas). Em resumo, os tumores podem causar:
- Dor de cabeça;
- Dificuldade de compreender e usar a linguagem;
- Perda de coordenação motora;
- Convulsões;
- Cansaço excessivo;
- Incapacidade de usar pernas ou braços;
- Paralisia; entre outros.

# 2. Guia de instalação
Se o objetivo for executar a aplicação sem o interesse de recriá-la, deve-se fazer o *build* da imagem Docker fornecida:

```bash
docker build -t deep-brain .
docker run deep-brain
```

Caso seu computar possua uma GPU dedicada, altere o `docker run` para habilitar o uso da GPU:

```bash
docker build -t deep-brain .
docker run -gpus all deep-brain
```


Para recriar o ambiente desenvolvimento em sua máquina, recomendamos a instalação do [Miniconda](https://docs.anaconda.com/miniconda/install/#quick-command-line-install).
Em seguida:

```bash
conda create -n deep-brain python==3.11
conda activate deep-brain
conda install pytorch torchvision torchaudio -c pytorch
conda install numpy pandas
```

Ou, em vez de realizar as instalações pelo conda (**recomendado**), pode-se utilizar o `pip` e instalar os pacotes diretamente do `requirements.txt`:

```bash
conda create -n deep-brain python==3.11
conda activate deep-brain
conda install pip
pip install -r requirements.txt
```
