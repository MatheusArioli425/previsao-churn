# Previsão de churn · Loja Horizonte

Modelo de **Machine Learning** que estima a chance de cada cliente parar de comprar, feito com **scikit-learn** (regressão logística).

> Os dados são fictícios, gerados para estudo. Uma versão interativa deste modelo está no meu portfólio.

## Problema

A loja perdia clientes e só percebia meses depois, quando já era tarde para agir. A pergunta: dá para saber **antes** quem está prestes a sair?

## O que eu fiz

1. **Explorei a base** de 2.000 clientes com 5 variáveis de comportamento: tempo como cliente, gasto mensal, reclamações abertas, dias desde a última compra e se usa o aplicativo. A taxa de churn é de 32,5%.
2. **Separei treino e teste:** 1.600 clientes para treinar e 400 que o modelo nunca viu para avaliar.
3. **Treinei uma regressão logística** com as variáveis padronizadas (`StandardScaler` + `LogisticRegression` em um `Pipeline`).
4. **Avaliei** com acurácia, precisão, recall, AUC e matriz de confusão.
5. **Ajustei o limite de decisão** pensando no negócio: para retenção, deixar um cliente escapar custa mais que um alarme falso.
6. **Interpretei os pesos** do modelo para dizer o que leva ao churn.

## Resultados

| Métrica (400 clientes de teste) | Limite 50% | Limite 40% |
|---|---:|---:|
| Acurácia | 79,5% | 78,2% |
| Recall (dos que saíram, quantos o modelo pegou) | 55,1% | 68,6% |
| Precisão (dos marcados como risco, quantos saíram) | 69,1% | 61,8% |

AUC de **0,84**.

- **Reclamações abertas e dias sem comprar** são os sinais mais fortes de saída.
- **Tempo de casa, gasto mensal e uso do app** seguram o cliente.
- Exemplo: um cliente com 4 meses de casa, 3 reclamações abertas e 90 dias sem comprar tem **95% de chance de churn**. Um cliente de 3 anos que usa o app tem 4%.

![Matriz de confusão](graficos/matriz_confusao.png)

![Pesos das variáveis](graficos/pesos.png)

## Como rodar

Você precisa do Python 3.10 ou mais novo.

```bash
pip install -r requirements.txt
jupyter notebook previsao_churn.ipynb
```

O notebook já está salvo com os resultados. Dá para ler direto aqui no GitHub.

## Estrutura

```
previsao_churn.ipynb   notebook com exploração, treino, avaliação e interpretação
dados/                 base de clientes em CSV
graficos/              matriz de confusão e pesos do modelo
requirements.txt       bibliotecas usadas
```

---
Feito por **Matheus Arioli** · Analista de Dados
