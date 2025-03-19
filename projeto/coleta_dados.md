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
