[![author](https://img.shields.io/badge/author-LeandroMinervino-red.svg)](https://www.linkedin.com/in/leandro-minervino-b469681b/) [![](https://img.shields.io/badge/python-3.7.12+-blue.svg)](https://www.python.org/downloads/release/python-365/) [![GPLv3 license](https://img.shields.io/badge/License-GPLv3-blue.svg)](http://perso.crans.org/besson/LICENSE.html) [![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)]()



# Avaliação de Risco de Crédito

![image](https://user-images.githubusercontent.com/48839817/160699385-61e674cb-c32c-43b4-94ea-80c9e08bd559.png)


# Introdução
Esse projeto é parte do curso Data Science na Prática e tem como intuito demonstrar uma abordagem prática de um problema real das empresas financeiras utilizando a biblioteca Pycaret como meio de criar um modelo de aprendizado de máquina.

O Nubank, uma fintech fundada em 2013, se encaixa nessa categoria. A Nubank criou uma competição de ML com o intuito de selecionar funcionários talentosos que conseguissem criar modelos preditivos sobre o *Default* dos clientes . Nesse projeto utilizarei o mesmo dataset utilizado na competição. Mostrarei o passo a passo da análise e transformação dos dados e posteriormente criarei o modelo de ML utilizando a biblioteca Pycaret com o intuito de demonstrar como essa biblioteca permite de maneira simples e eficiente analisar diversos modelos - permitindo selecionar aquele mais adequado ao problema.

# Dados

O dataset possui 45000 linhas e 43 colunas. A variável-alvo possui 35080 casos negativos e 6661 positivos, portanto desbalanceado. As colunas e seus tipos originais bem como a presença de nulos é como segue:

![image](https://user-images.githubusercontent.com/48839817/160701974-cf91e744-a0ad-45de-8986-c97201c60489.png)

As demais estatísticas descritivas podem ser verificadas no notebook

## Tratamento da base

Os seguintes passos foram feitos nessa etapa de limpeza de dados:

- Transformação dos valores infinitos em NaN.
- Remoção das fraudes positivas.
- Remoção dos dados criptografados.
- Remoção das linhas em que não há dados para a variável-alvo.
- Correção das colunas de e-mail de 999 para NaN e de erros de digitação.
- Utilizando o simple imputer com valor zero para as variáveis: 'last_amount_borrowed', 'last_borrowed_in_months' e 'n_bankruptcies'.
- Utilizando o simple imputer com a média da coluna para as variáveis: 'ok_since', 'external_data_provider_credit_checks_last_year', 'credit_limit', 'external_data_provider_email_seen_before', 'reported_income', 'n_defaulted_loans' e 'n_issues'.
- Utilizando o simple imputer com a moda da coluna para as variáveis: facebook_profile', 'marketing_channel' e 'lat_lon'.
- Correção da variável shipping_state - mantendo apenas a sigla do estado.
- Transformação da variável application_time_applied no formato hora
- Remoção da variável lat-long



## Machine Learning

Como utilizei o Pycaret para a seleção dos modelos criei um setup que todos os 4 melhores modelos compartilharam. A parte de normalização, balanceamento, remoção de outliers, feature selection, multicolinearidade foram todas tratadas diretamente pela biblioteca:

```
clf = setup(data =train,
            target = 'target_default',
            session_id=42,
            log_experiment=True,
            experiment_name='default',
            normalize=True,
            fix_imbalance = True,
            feature_selection = False,
           feature_selection_threshold = 0.8,
            remove_outliers = True,
            categorical_features = ['n_bankruptcies', 'n_defaulted_loans',
                                    'external_data_provider_credit_checks_last_month',
                                    'external_data_provider_credit_checks_last_year',
                                    'facebook_profile', 'email', 'marketing_channel',
                                    'shipping_state'],
            remove_multicollinearity = True,
            pca = True,
            ignore_low_variance = True
            )
  ```

Os modelos testados foram : Ridge Classifier, LDA, Logistic Regression, SVM. A métrica principal foi o *Recall*


## Conclusão

A escolha da métrica de avaliação de um modelo de aprendizado de máquina deve ser atrelada ao problema que se quer atacar.
Por se tratar de uma um projeto de detecção de *defaults* queremos minimizar a presença de falsos negativos - quando se trata de uma fraude mas classificamos como normal.
Por isso, a métrica principal que levei em conta na escolha dos modelos foi Recall:

![image](https://user-images.githubusercontent.com/48839817/148217331-d21d3dc7-5dd1-424a-a844-02fc3483e6f5.png)

Isso permitiu selecionar o SVM como melhor modelo com um Recall de 0,66.


Outras conclusões:

* Apesar do Pycaret facilitar o trabalho do cientista, ainda é preciso uma criteriosa análise exploratória e uma limpeza e seleção dos dados.
* A proximidade dos modelos em relação à métrica proposta nos obriga a treinar mais de um modelo - o que pode ser custoso em termos de capacidade computacional.
* Para um cientista inexperiente o uso do Pycaret pode tornar o modelo uma caixa-preta, já que os critérios são definidos diretamente pela biblioteca.
* O uso do Pycaret não exclui a possibilidade do cientista de dados trabalhar na criação e melhora dos seus modelos manualmente - por exemplo com feature engineering.


## Software & Bibliotecas

O Projeto utilizou Python 3 e as seguintes bibliotecas:

-   [Pandas](http://pandas.pydata.org/)
-   [Seaborn](https://seaborn.pydata.org/index.html)
-   [sklearn](https://scikit-learn.org/stable/)
-   [time](https://docs.python.org/3/library/time.html)
-   [numpy](https://numpy.org/)
-   [Pycaret](https://pycaret.gitbook.io/docs/)


**Meus Links:**

* [LinkedIn](https://www.linkedin.com/in/leandro-minervino-b469681b/)
* [Colab](https://colab.research.google.com/drive/1SkgqRVi2Y6016fKROhCkTVGpUivDyu3G?usp=sharing)



## Outros Projetos Meus:


* **[Scrap sites de notícia](https://github.com/leandrominer85/Scrap_sites_noticias)**

* **[Políticas Raciais no ensino superior](https://github.com/leandrominer85/Pol-tica-Racial-no-Ensino-Superior-2009-2019-)**
 
* **[Disaster-Response-Pipelines](https://github.com/leandrominer85/Disaster-Response-Pipelines)**

* **[Dados do Airbnb Lisboa](https://github.com/leandrominer85/Dados-do-Airbnb-Lisboa/blob/main/README.md)**

* **[Panorama COVID-19](https://github.com/leandrominer85/Panorama_Covid-19)**

* **[Detecção de Fraudes em Cartões de Crédito](https://github.com/leandrominer85/Deteccao_de_Fraude_em_Cartoes_de_Credito)**

