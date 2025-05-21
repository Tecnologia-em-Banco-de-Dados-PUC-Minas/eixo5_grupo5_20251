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

# 5. Preparação e pré-processamento dos Dados

Em ciência de dados e projetos de *machine learning* no geral, a etapa de preparação dos dados (*data preprocessing*) consiste no conjunto de operações cujo propósito é colocar o conjunto de dados em um formato adequado para o ajuste do modelo. Para dados tabulares, em via de regra, são realizadas as seguintes tarefas:

- Tratamento de qualidade (ainda que o cientista de dados realize esta etapa para certificar-se de que os dados estão em qualidade apropriada, é recomendado que os *pipelines* ETL/ELT projetados pelos engenheiros de dados já disponham de um processo robusto de controle de qualidade), como remoção de valores faltantes (caso não envolva regra de negócio), padronização de texto, etc.,
- Engenharia de variáveis (*feature engineering*), a qual consiste na criação de novas variáveis-preditoras a partir das existentes, além de escalonamento das variáveis numéricas (se necessário) e codificação das variáveis categóricas. Há muitas outras transformações possíveis (*features* polinomiais, codificação por diferentes estratégias, entre outras).

No caso de modelos de *computer vision*, como é o caso deste projeto, a etapa de processamento dos dados é um pouco diferente. O conjunto de dados não é, originalmente, uma tabela, e sim uma coleção de imagens. Estas imagens são convertidas em matrizes cujos elementos são a intensidade dos píxeis da imagem. Para este tipo de dado, realizamos etapas ligeiramente diferentes. Estas etapas são aplicadas tanto no conjunto de treino quanto de teste. São elas:

- Conversão da escala de cinza nos canais de cor: consiste na conversão da imagem *grayscale* nos canais RGB (vermelho, verde e azul). A arquitetura de rede convulacional (falaremos mais à frente sobre ela, na seção de **Desenvolvimento do modelo**) espera estes canais como *input*,
- Dimensionamento da imagem para o padrão mais comum de redes convulacionais (matriz quadrada 224x224),
- Espelhamento horizontal aleatório (*RandomHorizontalFlip*): esta técnica está contida no grupo de técnicas de enriquecimento de dados (*data augmentation*), bastante comum em projetos de visão computacional,
- Rotação aleatória: também consiste em uma técnica de enriquecimento em que a imagem é rotacionada aleatoriamente em certos graus (no nosso caso, 10),
- Conversão para a estrutura de tensor, exigida pelo PyTorch (biblioteca de aprendizado profundo que usaremos),
- Normalização: conversão dos valores dos píxeis de modo a formar uma distribuição de média 0 e variância unitária. Os valores de média e variância escolhidos são usados no projeto ImageNet.

É importante destacar que as redes convulacionais, assim como todos os modelos de aprendizado profundo no geral, fazem parte dos chamados modelos representacionais. Isto implica que técnicas de engenharia de variáveis não são estritamente necessárias, uma vez que o próprio modelo, atráves do ajuste dos pesos dos neurônios, é responsável por encontrar as *representações* que melhor correlacionam a variável-resposta com as variáveis-preditoras. Ainda assim, realizar as técnicas descritas ajuda o modelo a generalizar melhor (isto é, mitiga o *overfitting*). Por fim, vale notar que há muitas outras técnicas possíveis (como, por exemplo, o espelhamento vertical, que não foi usado) que poderiam ser exploradas.

O código que executa a etapa de pré-processamento dos dados está no *notebook* **development.ipynb** dentro da pasta "notebooks". Ao fim do trabalho, todo código contido no notebook será migrado para *scripts* Python de modo a obedecer as boas práticas de desenvolvimento de software. Optamos por escrever o código preliminarmente em notebooks pela facilidade de experimentação.

# 6. Desenvolvimento do modelo

A tarefa de classificar imagens associadas a rótulos pode ser realizada através de diferentes técnicas de aprendizado de máquina, no entanto, o método mais usado na academia e na indústria faz parte do chamado **aprendizado profundo** (*deep learning*). Esta técnica consiste no uso de redes neurais profundas, ou densas, as quais são formadas por diversas camadas ocultas (*hidden layers*) de neurônios entre a camada de entrada (*input layer*) e a camada de saída (*output layer*). O número de camadas na rede é um hiperparâmetro passível de ser otimizado, além do número de neurônios que cada camada terá. Os hiperparâmetros da rede são:

- Número de camadas,
- Número de neurônios,
- Taxa de aprendizado (*learning rate*),
- Função de ativação (é a função matemática de cada neurônio responsável por determinar se o neurônio deve ou não ser ativado, além de possibilitar o aprendizado de relações não lineares nos dados),
- Tamanho do batch,
- *Dropout rate* (a taxa que determina o "desligamento" de certos neurônios de maneira aleatória a fim de mitigar o sobreajuste, isto é, a perda na capacidade de generalização do aprendizado para novos dados),
- Otimizador (Adam, *Stochastic Gradient Descent* etc.)
- Método de inicialização dos pesos.

A partir de redes neurais "básicas", pode-se construir diferentes redes especializadas em determinadas tarefas. As redes neurais convolucionais são um tipo de arquitetura de rede neural profunda amplamente utilizada em tarefas de processamento de imagens, reconhecimento de padrões e visão computacional. Sua principal característica é a capacidade de extrair automaticamente características espaciais e hierárquicas dos dados por meio de camadas convolucionais.

O funcionamento das CNNs baseia-se na aplicação de filtros (ou *kernels*) que percorrem a imagem de entrada, realizando operações de convolução para detectar padrões locais, como bordas, texturas e formas. Cada filtro aprende a identificar uma característica específica durante o processo de treinamento. Após as camadas convolucionais, são frequentemente utilizadas camadas de pooling, que reduzem a dimensionalidade dos dados, preservando as informações mais relevantes e contribuindo para a generalização do modelo.

Ao final da rede, camadas totalmente conectadas (fully connected) são empregadas para realizar a classificação com base nas representações extraídas anteriormente. Devido à sua capacidade de capturar relações espaciais e aprender representações discriminativas, as CNNs se tornaram a abordagem padrão para diversas aplicações em análise de imagens e sinais visuais. 

## 6.1. Treino

O **Deep Brain** possui como *core* uma rede neural convolucional treinada em um *dataset* de imagens de ressonância magnética que contêm tumores cerebrais de diferentes tipos, além de imagens de cérebros saudáveis (class negativa), como explicado na primeira seção deste documento. O modelo foi treinado em um MacBook Air M1 que não possui GPU (placas gráficas), o que torna o treinamento do modelo bastante lento. As GPUs são particularmente eficientes no treino de redes neurais profundas por serem *hardware* especializado em operações matriciais, particularmente multiplicação de matrizes. O modelo demorou 1h30 para ser treinado. 

## 6.2. Seleção de modelo (otimização de hiperparâmetros)

Não realizamos nenhum tipo de otimização de hiperparâmetro devido à ausência de GPU, o que tornaria todo o processo de treino extremamente demorado (potencialmente levando um dia ou mais até todo o processo terminar). No entanto, é importante destacar que o *hyperparameter tuning* é um processo corriqueiro e extremamente recomendado por possibilitar o aumento considerável de performance e a mitigação do sobreajuste. Há vários métodos para realizá-lo, mas 3 se destacam:

- **GridSearch**: consiste na definição de um espaço de hiperparâmetros e no ajuste do modelo utilizando cada uma das combinações possíveis. Ao final, escolhe-se a seleção que maximiza o a métrica de avaliação escolhida para o modelo. É um método clássico mas potencialmente problemático, uma vez que, se o espaço de escolha for demasiado grande, o número de combinações "explodirá" e o processo pode demandar dezenas de horas ou dias, a depender do hardware e da rede que se quer treinar. O ponto positivo é que se explora todas as combinações possíveis de hiperparâmetro,
- **RandomSearch**: similar ao método acima, mas a escolha de hiperparâmetros é feita de maneira randômica, encurtando o tempo de otimização mas sem a garantia de que a melhor combinação será encontrada,
- **Otimização Bayesiana**: esta técnica vale-se do teorema de Bayes para explorar um espaço de hiperparâmetro, combinada com um processo de otimização (minimização de uma *loss*). Ela utiliza modelos probabilísticos, como processos Gaussianos ou árvores de regressão, para modelar a função de desempenho do modelo e selecionar novos pontos promissores com base em critérios de aquisição relacionados à metrica de modelagem. É um dos métodos mais utilizados pois, apesar de não garantir a convergência para a seleção ótima, produz resultados *near optimal* em tempo razoável.

## 6.3. Redes pretreinadas (*transfer learning*)

Além do desenvolvimento de uma rede convolucinal do zero, poder-se-ia ter optado pelo uso de uma rede pré-treinada, o que é mais comum de se observar na academia e na indústria. Estas redes foram previamente treinadas em *datasets* massivos de imagens e podem ser reutilizadas para uma tarefa específica que se queira executar (como a classificação de imagens médicas). O benefício do uso de redes deste tipo é a possibilidade de detectar relações nos dados bastante sutis, além de economizar em tempo de treinamento do modelo. A técnica de transferir o aprendizado em um domínio para outro chamamos de transferência de aprendizado, ou *transfer learning*. Há dois métodos possíveis para isto: 

- **Extração de features**: todas as camadas convolucionais ficam congeladas e apenas as camadas totalmente conectadas (*fully connected layers*), chamadas de *classification head*, são atualizadas a partir do novo conjunto de dados. A vantagem desta técnica é a velocidade e o menor risco de sobreajuste, com a desvantagem da perda de relações mais sutis e de baixo nível entre os dados. No geral, é melhor que o novo conjunto de dados seja similar que o conjunto de treino da rede,
- **Fine-tuning**: neste caso, todas a camadas (mas não *necessariamente* todas) podem ser atualizadas, desde as totalmente conectadas até as convolucionais. "Atualizar" uma camada nada mais é do que atualizar o peso de cada neurônio. A vantagem é a possibilidade de aprender nuances nos dados e obter um modelo mais eficiente na tarefa, mas a exigência de tempo e de volume de dados é bastante maior.

# 7. Avaliação de resultados

## 7.1. Matriz de confusão

Em problemas de classificação, a matriz de confusão relaciona os desfechos observados (empíricos, reais) e os estimados pelo modelo estatístico. Através dela é possível computar os erros de tipos I e II, além de analisar a performance geral do classificador. A matriz de confusão é a base para o cálculo de muitas outras métricas de avaliação de modelos estatísticos, como veremos abaixo.

## 7.2. *Precision, recall* e *F1-score*

Para avaliar os resultados, plotamos a curva que relaciona a *loss* em cada época de treino (as perdas devem diminuir progressivamente, ainda que pequenas flutuações sejam possíveis de uma época par aoutra) para avaliar o processo de treino do modelo e, posteriormente, visualizamos algumas métricas de classificação através do objeto ```classification report``` da biblioteca *Scikit-learn* e da plotagem das curvas *Precision-Recall* para cada classe (tipo de tumor) junto de suas linhas iso-F1.

A curva PR relaciona *scores* de precision e recall (ou sensibilidade). A *precision* consiste na razão entre previsões da classe positiva (ou seja, prever que uma imagem contém um tumor) e os acertos destas previsões. Por exemplo, se o modelo realizar 10 previsões afirmando a presença de tumor mas apenas 4 das imagens as contiver, a precision será de 40%. A sensibilidade (*recall*), por outro lado, mede o quanto de observações da classe positiva realmente foram capturadas. Se o modelo foi capaz de prever corretamente a presença de tumor em 6 imagens de tumor em um conjunto que contenha 10, então a sensibilidade é de 60%. Matematicamente, a *precision* é definida por:

$$
\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}
$$

Em que TP = *True Positives* = Verdadeiros Positivos e FP = *False Positives* = Falsos Positivos. A *precision* controla a taxa do erro tipo I, isto é, rejeitar a hipótese nula quando não se tem evidências suficientes para fazê-lo (nota-se que, tecnicamente, é incorreto dizer *aceitar* ou *provar* a hipótese nula). Esta métrica está totalmente relacionada com a taxa de falso positivo (alpha). Dito de uma forma mais simples, sobretudo num contexto de aprendizado de máquina, a *precision* controla a taxa na qual o modelo resulta em classificação positiva para tumor em uma imagem de um cérebro saudável. Já a sensibilidade é matematicamente descrita pela seguinte equação:

$$
\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}
$$

Em que FN = *False Negatives* = Falsos Negativos. Neste caso, consiste em não rejeitar a hipótese nula quando há evidências para fazê-lo. No nosso contexto, significa que o modelo falhou em detectar uma imagem que contém tumor, classiicando-a como saudável. Em modelagem, a etapa de definição dos **custos** de cada erro é extremamente importante para balizar a ênfase que se deve dar no modelo. Por exemplo, em um cenário de marketing, o custo de oferecer um produto para um cliente que está fora das regras de negócio não é tão grande - no máximo um cliente comprará um produto que não foi feito para o público ao qual pertence. Ou então realizar comunicação a um cliente que possui determinado cartão de crédito, com base em um modelo de *churn* (cancelamento do cartão) que estima alta probabilidade de cancelamento para um cliente que não ia, de fato, cancelar - pode-se gerar atrito, mas nada grave acontecerá. No entanto, no contexto clínico, pode haver consequências bastante graves para cada tipo de erro. 

A classificação incorreta da presença de uma doença em um determinado paciente pode trazer transtornos ao demandar novos exames para assegurar de que o resultado é verídico ou ao obrigar que o paciente permaneça no hospital (no caso de doenças infecto-contagiosas), mas, via de regra, é preferível do que a ocorrência de falsos negativos, momento no qual um paciente com uma doença altamente contagiosa e/ou potencialmente grave volta para casa com a falsa informação de que é saudável. 

Como esta solução trata de um modelo de classificação de tumor, a taxa de falso negativo deve ser a menor possível, sob risco de dizer a um paciente que ele é saudável quando, na realidade, não é. Isto o leva a perder tempo no tratamento do tumor o que, em oncologia, é extramente sério. Portanto, deseja-se maximizar a sensibilidade do modelo. Ao mesmo tempo, não se pode perder de vista que um modelo com sensibilidade extremamente alta mas precisão baixa é essencialmente irrelevante. No limite, para maximizar a sensibilidade, basta classificar todos, ou quase todos, os pacientes como portadores de um tumor, o que é, evidentemente, absurdo. Assim, é necessário que a previsão do modelo seja bastante confiável e que, além disso, ele seja sensível à presença de tumor. É por isso que testes diagnósticos costumam ter tanto a *precision* quanto o *recall* em valores extremamente altos (ex: 99% ou 99.9%). 

Uma métrica que balanceia estas duas outras é o F1-score. Ele consiste em uma média harmônica entre precisão e sensibilidade. Um ponto negativo é que um mesmo valor de F1-score pode ser obtido através de valores muito diferentes de precisão e sensibilidade. Se a precisão e a sensibilidade tiverem o mesmo valor, ou forem muito próximos, então o F1-score terá um valor também próximo ou igual a estas métricas. Se, no entanto, a precisão estiver muito distante da sensibilidade (e vice-versa), poderíamos obter o mesmo valor de F1 em relação ao anterior (ou seja, se, em ambos os exemplos, a precisão e a sensibilidade tiverem valores tais que a média harmônica entre eles resulta no mesmo valor). Outro ponto desfavorável da métrica F1 é que ela não incorpora o tamanho das classes, ou seja, o desbalanceamento em um conjunto de dados não é levado em consideração, o que pode levar a uma interpretação exageradamente otimista sobre um classificador.

É importante destacar que a precisão e a sensibilidade costumam caminhar em sentido oposto, isto é: quanto maior a precisão, menor tende a ser o recall.

## 7.3. Coeficiente de correlação de Matthews (do inglês *Matthews' correlation coefficient*)

Esta métrica (MCC) não é frequentemente utilizada na indústria, mas é bastante relevante no contexto clínico e de bioestatística. O coeficiente de correlação de Matthews mede o quanto um estimador está associado aos desfechos observados. Em problemas de classificação binária, reduz-se ao coeficiente de correlação de Pearson. A vantagem de se utilizar o MCC é que ela incorpora o desbalanceamento das classes, removendo o falso otimismo que o score F1 pode oferecer. O valor desta métrica está contido no intervalo ```[-1, 1]```, em que -1 indica que as previsões e o desfecho observado caminham em sentido posto (negativamente associados ou correlacionados), 0 indica correlação idêntica ao aleatório e +1 indica correlação perfeita.

## 7.4. Resultados obtidos na modelagem

Foi necessária 1h45min para o treino do modelo em 10 épocas. Os resultados foram os seguintes:

Gráfico 1: Matriz de confusão.

![Matriz de confusão](/projeto/docs/matriz_confusao.png)

Tabela 1: Performance do modelo.

| Class | Precision | Recall | F1-score |
| ------ | --------- | ------- | ------ |
| glioma  | 0.92 | 0.63 | 0.74 |
| meningioma  | 0.64 | 0.68 | 0.66 |
| pituitary  | 0.93 | 0.96 | 0.94 |
| notumor  | 0.85 | 0.99 | 0.91 |

Gráfico 2: Curvas *Precision-Recall* e linhas iso-F1.

![Curvas PR e linhas iso-F1](/projeto/docs/curvas_pr.png)

Coeficiente de correlação de Matthews do modelo: 77.1%.

Com base nos resultados obtidos, observa-se que o modelo foi capaz de aprender a realizar a tarefa desejada, se saindo relativamente bem na detecção de todos os tipos de tumores apresentados. Há, todavia, possíveis ressalvas com relação ao meningioma e, em menor grau, ao glioma. O modelo parece ter dificuldade na detecção do meningioma, ainda que sua frequência seja praticamente a mesma dos outros tipos de tumor. Quanto ao glioma, a precisão do modelo é bastante grande, mas há uma leve perda em sensibilidade com relação à performance na detecção do tumor na glândula pituitária. Quando o modelo classifica alguém como saudável, podemos ter razoável confiança de que a pessoa é, de fato, saudável (para cada 100 previsões de ausência de câncer, o modelo acerta em 85 delas e consgue abranger 91% de todas as imagens que não possuem tumor).

O ranqueamento das classes em termos de performance é: 

1. Pituitário,
2. Glioma,
3. Meningioma,
4. Ausência de tumor.

## 7.5. Possíveis melhorias

- Aumento das épocas de treino (não exige nenhuma mudança nas configurações do modelo em si),
- Otimização de hiperparâmetros via busca Bayesiana (implementada pelo pacote *HyperOpt*),
- Utilização de uma rede pré-treinada e *transfer learning* para aumento de performance, sobretudo no caso do glioma.

## 7.6. Utilização

No cenário hipotético do projeto (ambiente hospitalar), recomendaríamos que o modelo fosse utilizado como suporte à tomada de decisão em equipe multidisciplinar constituída de médico oncologista, físico médico e radiologista. **Para investigações relativas ao tumor do tipo glioma, não se recomenda o uso**.