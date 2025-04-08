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
