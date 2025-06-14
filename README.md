
# Segmentação de doenças em plantas usando U-Net
A rede neural convolucional U-Net tem como objetivo a segmentação precisa de imagens de alta resolução, sendo bastante utilizada na área médica para identificar tumores, células e órgãos. Devido à sua precisão, essa rede também se mostra eficaz para identificar áreas afetadas por doenças em folhas de plantas, o que permite agir mais cedo para evitar prejuízos na produção agrícola.  Utilizando a base de dados PlantVillage, o modelo foi treinado e avaliado com foco em precisão pixel a pixel. A versão modificada da rede apresentou desempenho superior em métricas como IoU e BCE Loss, demonstrando como ferramentas computacionais servem de suporte à agricultura de precisão.

Este repositório contém a implementação completa de três arquiteturas de redes neurais aplicadas à segmentação de imagens da base PlantVillage:

- Autoencoder Convolucional
- U-Net
- ResUNet

## Alunos integrantes da equipe

* Gabrielle Lira Dantas Wanderley
* Henrique Santana Diniz
* Joaquim de Moura Thomaz Neto
* Maria Eduarda Chrispim Santana
* Saulo Pereira Vasconcelos

## Professor responsável

- Leonardo Vilela Cardoso

## Instruções de Uso

### Requisitos

- Python 3.8 ou superior
- [Anaconda (recomendado)](https://www.anaconda.com/)
- PyTorch, torchvision, PIL, matplotlib

### Instalação de Dependências

Crie e ative um ambiente virtual (recomendado com `conda`):

```bash
conda create -n segmentacao_folhas python=3.10
conda activate segmentacao_folhas
pip install -r requirements.txt
```

### Estrutura do Projeto

Cada arquitetura possui sua própria pasta com o código-fonte, imagens e resultados organizados separadamente:

```
├── Autoencoder/
│   ├── train.py
│   ├── gerar_mascaras.py
│   ├── autoencoder.py
│   ├── imagens/
│   ├── mascaras/
│   └── resultados/
│
├── UNet/
│   ├── train.py
│   ├── gerar_mascaras.py
│   ├── unet.py
│   ├── imagens/
│   ├── mascaras/
│   └── resultados/
│
├── ResUNet/
│   ├── train.py
│   ├── gerar_mascaras.py
│   ├── resunet.py
│   ├── imagens/
│   ├── mascaras/
│   └── resultados/
```

### Execução

Para executar o treinamento de qualquer uma das arquiteturas, siga os passos abaixo:

1. Navegue até a pasta da arquitetura desejada (por exemplo, `Autoencoder/`).
2. Execute o script de treinamento:

```bash
python train.py
```

As predições geradas após o treinamento serão salvas automaticamente na subpasta `resultados/`.

### Criação de Máscaras

As máscaras podem ser geradas a partir das imagens utilizando o script `gerar_mascaras.py`. Esse script permite que o usuário selecione manualmente as regiões afetadas por doença:

1. Coloque as imagens na pasta `imagens/`.
2. Execute o script:

```bash
python gerar_mascaras.py
```

3. Abre imagens na pasta imagens/ e permite que o usuário desenhe as regiões doentes com o mouse. Ao apertar s, a máscara é salva automaticamente na pasta mascaras/.
4. Comandos:

s → Salva a máscara desenhada

q → Pula imagem sem salvar

As máscaras são imagens em preto e branco, onde os pixels marcados com a área da doença aparecem em branco (255) e o restante em preto (0).

### Observações

- As imagens de entrada e as máscaras devem estar nas pastas `imagens/` e `mascaras/`, com os **mesmos nomes de arquivos**.
- Os modelos produzem imagens reais e previstas para análise visual após o treinamento.
