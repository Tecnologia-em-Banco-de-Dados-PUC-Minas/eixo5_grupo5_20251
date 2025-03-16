![DeepBrain Logo](/projeto/docs/blendBoard.png)

# 1. Introdução
### 1.1 Panorama do projeto
O **Deep Brain** é uma aplicação de machine learning capaz de detectar tumores (glioma, meningioma e pituitário) em imagens de ressonância magnética do cérebro. 
A aplicação é baseada em uma rede neural convulacional treinada em imagens destes tipos de tumores. Esta rede retorna probabilidades associadas a cada tipo de tumor - ou à ausência de tumor - em uma imagem de ressonância fornecida pelo usuário. A motivação do projeto consiste em auxiliar e acelerar o processo diagnóstico deste tipo de patologia, fornecendo um eixo
a mais na tomada de decisão de profissionais da saúde que atuam na área de oncologia. 

### 1.2. Validação do modelo
Para validar o modelo subjacente à aplicação, utilizamos técnicas estatísticas e métricas mais adequadas ao contexto de saúde - em particular, a um caso de uso em que **o falso negativo é mais custoso do que o falso positivo**. Isto acontece porque, ao classificar um paciente como negativo (não possui tumor), o modelo impede que o paciente receba tratamento em tempo hábil, impactando a probabilidade de sobrevivência (no caso de tumores malignos) e causando dano desnecessário. As métricas que relacionam falsos positivo e negativo são a precisão (*precision*) e o *recall* (sensibilidade), respectivamente. De forma simples, a precisão mede a proporção de previsões corretas em relação ao total de previsões feitas. Ela é uma métrica impactada por falsos positivos pois um valor baixo de precisão indica que muitas previsões foram feitas, mas poucas foram corretas (indicando alta taxa de falso positivo). A sensibilidade, por outro lado, mostra a proporção de imagens contendo tumores que o modelo corretamente encontrou - neste caso, um *score* baixo de sensibilidade indica que o modelo é incapaz de encontrar a maioria das imagens contendo tumores (classificando muitas imagens como "saudáveis"). Isto implica em uma taxa alta de falso negativo. 

Além das métricas isoladas de precisão e sensibilidade, usamos também o F2-*score* (obtido pela equação do Fβ-*score*), que consiste em uma média entre precisão e sensibilidade mas que atribui um peso maior a esta última, e o **coeficiente de correlação de Mathews (MCC)**. Esta métrica é bastante utilizada em biologia computacional e bioestatística, pois considera toda a matriz de confusão no cômputo do *score*, englobando os desbalanceamentos entre as classes. Discutiremos mais adiante como o MCC é usado para revelar que supostos bons classificadores (na ótica dos *scores* F1 e F2) podem ser, na verdade, bastante ruins. Assim, nosso principal balizador da qualidade do modelo será o MCC e, em menor peso, o *F2-score*. Como este projeto se trata de um classificador multi-classes, as métricas de precisão, sensibilidade e F2 são ajustadas de acordo (precisamos usar estratégias de agregação; optamos pelo ponderamento das 
classes)

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

# 3. Governança de Dados
Este projeto se trata de uma simulação do ambiente de tecnologia da informação de um hospital - mais especificamente, da área encarregada por projetos de dados. Neste contexto, um projeto de governança de dados requer cuidados específicos relativos à área médica, uma vez que envolve dados PII (*personally identifiable information*, ou seja, dados pessoais identificáveis) e dados biométricos. Como o escopo deste trabalho é limitado ao nível de uma tarefa acadêmica, ele não engloba a totalidade de dados que se encontraria em uma prática real. Sendo assim, não é possível *implementar* um projeto de governança. No entanto, é viável demonstrar um protótipo desta estrutura de governança de dados. Se fôssemos realizar esta implementação, a governança de dados do Deep Brain partiria dos seguintes pilares:

## 3.1. Identificação e Classificação de Dados

- Dados PII: nome, registro nacional (CPF, RG ou RNE em caso de estrangeiros), telefone e outras informações que possam identificar diretamente o paciente;

- Resultados de exames: englobam dados biométricos, envolvendo, potencialmente, dados genéticos, diagnósticos e perícias médicas;

- Imagens de ressonância magnética: são os dados principais deste projeto e serão os registros utilizados pelo Deep Brain para detecção de tumores;

- Identificador único do paciente, "ID_paciente", que será utilizado para correlacionar seus dados sem expor informações pessoais diretamente. Também será utilizado como chave primária para cruzamento de informações no modelo de dados.

## 3.2. Anonimização e Proteção de Dados

Para garantir a segurança dos pacientes e o cumprimento das regulações de privacidade, os seguintes princípios serão adotados:

- Anonimização dos dados PII: os dados identificáveis serão *hasheados* para garantir sua anonimização. O **ID_paciente** será utilizado para correlacionar exames e imagens sem revelar a identidade do paciente;

- Criptografia: dados PII e exames armazenados serão protegidos com criptografia em repouso e em trânsito, garantindo que apenas usuários autorizados tenham acesso. Possíveis serviços de armazenamento em nuvem (o que é permitido para dados médicos segundo a LGPD) permitem a criptografia dos dados, como o **AWS RDB** ou **AWS S3**;

- Controle de acesso: implementação de permissões baseadas em papéis (*role-based access control*) para garantir que apenas profissionais autorizados tenham acesso aos diferentes tipos de dados.

## 3.3. Qualidade e Integridade dos Dados

Para assegurar a precisão e confiabilidade dos dados médicos, serão adotadas as seguintes práticas:

- Padronização dos dados: definição de formatos consistentes para imagens, metadados, dados médicos diversos e dados PII. As imagens utilizarão formato .JPG. Dados tabulares (metadados, dados médicos do tipo estruturado e dados PII) serão armazenados no formato **parquet**. O formato **parquet** é, geralmente, utilizado em *Big Data* por ser mais eficiente para alto volume de dados. Obviamente, não é o caso deste projeto, mas, em um contexto hospitalar, é um cenário bastante factível. A etapa de padronização é realizada nos *pipelines* ETL (*Extract, Transform and Load*) e são de responsabilidade - em áreas maduras de dados - do engenheiro de dados;

- Validação e auditoria: implementação de processos automáticos e revisões periódicas para detectar inconsistências ou anomalias nos dados. Esta etapa diz respeito ao controle de qualidade dos dados. Uma vez que as bases são implementadas em produção, elas devem ser monitoradas para garantir consistência e para possíveis processos de auditoria;

## 3.4. Segurança e Compliance

Um dos aspectos mais importantes em uma organização é a aderência jurídica ao tratamento, armazenamento e distribuição de dados. No caso hospitalar, há o armazenamento e tratamento de dados biométricos, genéticos e sensíveis, os quais requerem considerações legais específicas. O projeto seguirá regulamentações locais e internacionais sobre privacidade e proteção de dados.

- Aderência à **Lei Geral de Proteção de Dados (LGPD)**: esta é a norma jurídica brasileira mais importante para conformidade legal no tratamento de dados. Ela dispoõe o que são dados pessoais, dados sensíveis e dados identificáveis, além dos requisitos jurídicos para armazenamento, tratamento e distribuição de dados de cidadãos brasileiros;

- Logs de auditoria: registros de acessos e modificações nos dados serão mantidos para garantir rastreabilidade;

- Consentimento informado (*informed consent*): pacientes serão informados sobre o uso de seus dados e deverão consentir explicitamente antes de participarem de sistemas de decisão baseados em *machine learning*;

- Criação de um Comitê de Ética e Privacidade para supervisionar o uso adequado dos dados e garantir que não haja exposição indevida das informações dos pacientes.

## 3.5. Armazenamento e Retenção de Dados

- Período de retenção: Os dados serão armazenados pelo período mínimo necessário para treinamento e aprimoramento do modelo, respeitando as normativas locais. Neste caso, soluções de armazenamento em nuvem, como o **AWS S3**, também permitem o controle da temporalidade dos dados pelo gerenciamento do ciclo de vida. Esta funcionalidade é fundamental para automatizar práticas de governença a fim de garantir conformidade legal e ética à LGPD;

- Descarte seguro: ao fim do período de retenção, os dados serão descartados de forma segura por meio de técnicas de destruição irreversível.

# 4. Proposta de arquitetura da solução

![Arquitetura da solução](/projeto/docs/arquitetura-solucao-deep-brain.png)

## 4.1. Implementação do pilar de segurança e compliance

- AWS IAM (Identity & Access Management) – Controle granular de acesso a dados médicos sensíveis.
- AWS KMS (Key Management Service) – Para criptografia de dados sensíveis em repouso e em trânsito.
- Amazon CloudWatch – Para monitoramento da aplicação e alertas de segurança.
- AWS Config & AWS Audit Manager – Para auditoria e conformidade com regulamentos como LGPD.