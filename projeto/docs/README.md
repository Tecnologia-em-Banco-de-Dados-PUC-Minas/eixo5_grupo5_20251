![DeepBrain Logo](/projeto/docs/blendBoard.png)

# 1. Introdução
### 1.1 Panorama do projeto
O **Deep Brain** é uma aplicação de machine learning capaz de detectar tumores (glioma, meningioma e pituitário) em imagens de ressonância magnética do cérebro. 
A aplicação é baseada em uma rede neural convulacional treinada em imagens destes tipos de tumores. Esta rede retorna probabilidades associadas a cada tipo de tumor - ou à ausência de tumor - em uma imagem de ressonância fornecida pelo usuário. A motivação do projeto consiste em auxiliar e acelerar o processo diagnóstico deste tipo de patologia, fornecendo um eixo
a mais na tomada de decisão de profissionais da saúde que atuam na área de oncologia. 

### 1.2. Validação do modelo
Para validar o modelo subjacente à aplicação, utilizamos técnicas estatísticas e métricas mais adequadas ao contexto de saúde - em particular, a um caso de uso em que **o falso negativo é mais custoso do que o falso positivo**. Isto acontece porque, ao classificar um paciente como negativo (não possui tumor), o modelo impede que o paciente receba tratamento em tempo hábil, impactando a probabilidade de sobrevivência (no caso de tumores malignos) e causando dano desnecessário. As métricas que relacionam falsos positivo e negativo são a precisão (*precision*) e o *recall* (sensibilidade), respectivamente. De forma simples, a precisão mede a proporção de previsões corretas em relação ao total de previsões feitas. Ela é uma métrica impactada por falsos positivos pois um valor baixo de precisão indica que muitas previsões foram feitas, mas poucas foram corretas (indicando alta taxa de falso positivo). A sensibilidade, por outro lado, mostra a proporção de imagens contendo tumores que o modelo corretamente encontrou - neste caso, um *score* baixo de sensibilidade indica que o modelo é incapaz de encontrar a maioria das imagens contendo tumores (classificando muitas imagens como "saudáveis"). Isto implica em uma taxa alta de falso negativo. 
Além das métricas isoladas de precisão e sensibilidade, usamos também o F2-*score* (obtido pela equação do Fβ-*score*), que consiste em uma média entre precisão e sensibilidade mas que atribui um peso maior a esta última, e o **coeficiente de correlação de Mathews (MCC)**. Esta métrica é bastante utilizada em biologia computacional e bioestatística, pois considera toda a matriz de confusão no cômputo do *score*, englobando os desbalanceamentos entre as classes. Discutiremos mais adiante como o MCC é usado para revelar que supostos bons classificadores (na ótica dos *scores* F1 e F2) podem ser, na verdade, bastante ruins. Assim, nosso principal balizador da qualidade do modelo será o MCC e, em menor peso, o *F2-score*. Como este projeto se trata de um classificador multi-classes, as métricas de precisão, sensibilidade e F2 são ajustadas de acordo (precisamos usar estratégias de agregação; optamos pelo ponderamento das classes)
No que diz respeito a técnicas de avaliação, usamos: 
- Curvas de precisão-sensibilidade associadas às suas áreas (para cada classe) e as curvas iso-F2;
- Método de Monte Carlo para estimação da distribuição do *score*-F2 e MCC;
- Matriz de confusão.

### 1.3. Sobre tumores cerebrais
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
