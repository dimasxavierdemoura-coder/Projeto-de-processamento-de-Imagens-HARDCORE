# Processamento de Imagens - Projeto HARDCORE

Este projeto entrega um pipeline completo de visão computacional nível HARDCORE, com suporte a imagens coloridas, extração avançada de features, machine learning tradicional e deep learning com CNNs, validação de modelos, balanceamento de dataset, data augmentation, otimização de hiperparâmetros e benchmark comparativo.
📖 **Para descrição completa do projeto, tecnologias e arquitetura detalhada, consulte [PROJECT_DESCRIPTION.md](PROJECT_DESCRIPTION.md)**
## 🚀 Recursos Implementados

### Processamento de Imagens
- ✅ **Suporte a imagens coloridas** (RGB) e escala de cinza
- ✅ **Filtros de borda**: Sobel, Prewitt, Laplacian, Canny
- ✅ **Operações morfológicas**: Dilatação, Erosão, Abertura, Fechamento, Hit-or-Miss
- ✅ **Histogramas de cor**: Distribuição RGB para análise de cores
- ✅ **Features de textura**: LBP (Local Binary Patterns), HOG (Histogram of Oriented Gradients)
- ✅ **Features de forma**: Momentos de Hu, área, perímetro, circularidade

### Machine Learning
- ✅ **RandomForest**: Classificação tradicional com features extraídas
- ✅ **CNN 1D**: Redes neurais convolucionais em features
- ✅ **CNN 2D**: Redes neurais convolucionais diretamente em imagens
- ✅ **Data Augmentation**: Rotação, flip, zoom, shift para aumentar dataset
- ✅ **Balanceamento**: Oversampling com RandomOverSampler para classes desbalanceadas
- ✅ **Grid Search**: Otimização de hiperparâmetros para CNNs
- ✅ **Benchmark**: Comparação RF vs CNN em mesmo dataset
- ✅ **Validação**: Separação de conjunto de teste, métricas de avaliação
- ✅ **Inferência**: Predição em novas imagens com modelos treinados

### Infraestrutura
- ✅ **Dataset Inspector**: Validação e limpeza de datasets
- ✅ **Notebook Interativo**: Experimentos e visualização em Jupyter
- ✅ **CLI Completa**: Interface de linha de comando para todos os modos
- ✅ **Persistência de Modelos**: Salvamento e carregamento de modelos treinados

## 📁 Estrutura do Projeto
```
processamento de imagens/
├── image_processing.py      # Processamento de imagens
├── ml_pipeline.py          # Pipeline de ML
├── demo.py                 # CLI principal
├── dataset_inspector.py    # Inspeção de datasets
├── experiments.ipynb       # Notebook interativo
├── start.bat              # Instalação automatizada
├── clean_for_github.bat   # Limpeza para GitHub
├── .gitignore             # Arquivos ignorados
├── requirements.txt       # Dependências
├── README.md              # Documentação
└── PROJECT_DESCRIPTION.md # Descrição completa
```

## Instalação Automática

Para uma instalação completamente automatizada, execute o arquivo `start.bat`:

```bash
start.bat
```

Este script irá:
1. Verificar se Python está instalado
2. Criar um ambiente virtual
3. Instalar todas as dependências automaticamente
4. Mostrar instruções de uso

**Importante**: Antes de executar o `start.bat`, abasteça a pasta `dataset/` com suas imagens organizadas em subpastas por classe.

## Instalação Manual

1. Crie um ambiente Python:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. Instale dependências:
   ```bash
   pip install -r requirements.txt
   ```

## Uso

### 1. Processar bordas e salvar resultados
```bash
python demo.py --mode edges --input caminho/para/imagem.jpg --output output
```

### 2. Aplicar operações morfológicas
```bash
python demo.py --mode morphology --input caminho/para/imagem.jpg --output output --thresh 140
```

### 3. Extrair features de uma imagem
```bash
python demo.py --mode features --input caminho/para/imagem.jpg --output output
```

### 4. Treinar um classificador (RandomForest)
```bash
python demo.py --mode train --dataset dataset/rotulado --output output --model-out model.joblib --model-type rf
```

### 5. Treinar CNN 2D com balanceamento e augmentation
```bash
python demo.py --mode train --dataset dataset/rotulado --output output --model-out model_cnn.joblib --model-type cnn_2d --balance --augment
```

### 6. Executar benchmark RF vs CNN
```bash
python demo.py --mode benchmark --dataset dataset/rotulado --output output --balance --augment
```

### 7. Otimizar hiperparâmetros da CNN
```bash
python demo.py --mode grid-search --dataset dataset/rotulado --output output --balance --augment
```

### 8. Fazer predição em uma nova imagem
```bash
python demo.py --mode predict --input caminho/para/imagem.jpg --model-in model.joblib --output output
```

## Resultados dos Experimentos

### Benchmark RF vs CNN 2D
- **RandomForest**: 82.50% accuracy, 82.46% macro F1
- **CNN 2D**: 82.50% accuracy, 82.14% macro F1
- **Vencedor**: CNN 2D (empate técnico)

### Grid Search CNN
Melhores parâmetros encontrados:
- learning_rate: 0.001
- batch_size: 16
- epochs: 20

### Validação do Modelo
- Conjunto de teste separado (20% dos dados)
- Métricas: accuracy, precision, recall, F1-score
- Matriz de confusão gerada automaticamente

## Tecnologias Utilizadas

### 🐍 **Linguagem Principal**
- **Python 3.10+**: Linguagem de programação principal, conhecida por sua simplicidade e vasto ecossistema de bibliotecas científicas

### 🖼️ **Processamento de Imagens**
- **OpenCV 4.9.0**: Biblioteca open-source para visão computacional, usada para carregamento de imagens, filtros de borda (Sobel, Canny, Laplacian), operações morfológicas (dilatação, erosão, abertura, fechamento) e manipulação básica de imagens
- **scikit-image 0.25.2**: Biblioteca especializada em processamento de imagens, utilizada para extração avançada de features como HOG (Histogram of Oriented Gradients) e LBP (Local Binary Patterns)

### 🤖 **Machine Learning e Deep Learning**
- **scikit-learn 1.7.2**: Biblioteca fundamental para machine learning tradicional, incluindo RandomForest para classificação, métricas de avaliação (accuracy, F1-score, precision, recall) e ferramentas de validação cruzada
- **TensorFlow 2.x**: Framework de deep learning para construção e treinamento de redes neurais, usado para implementar CNNs (Convolutional Neural Networks)
- **Keras**: API de alto nível integrada ao TensorFlow, simplifica a criação de modelos de deep learning com camadas Conv2D, MaxPooling, Dense e Dropout

### ⚖️ **Balanceamento de Dados**
- **imbalanced-learn**: Biblioteca especializada em técnicas de balanceamento de datasets desbalanceados, utilizando RandomOverSampler para oversampling de classes minoritárias

### 📊 **Análise de Dados e Visualização**
- **NumPy**: Biblioteca fundamental para computação numérica, operações com arrays multidimensionais e matemática avançada
- **Pandas**: Biblioteca para manipulação e análise de dados estruturados, usada para criar DataFrames com features extraídas
- **Matplotlib**: Biblioteca para criação de gráficos e visualizações, integrada ao Seaborn para plots mais avançados
- **Seaborn**: Biblioteca de visualização estatística baseada no Matplotlib, usada para gráficos de avaliação de modelos

### 🔧 **Infraestrutura e Utilitários**
- **joblib**: Biblioteca para serialização de objetos Python, usada para salvar e carregar modelos treinados (.joblib files)
- **Pillow (PIL)**: Biblioteca para processamento básico de imagens, integrada ao OpenCV para operações complementares
- **scikit-learn preprocessing**: Módulos para pré-processamento de dados, incluindo LabelEncoder para codificação de labels categóricos

### 🖥️ **Ambiente de Desenvolvimento**
- **Jupyter Notebook**: Ambiente interativo para experimentação, visualização e documentação de código
- **Virtual Environment (.venv)**: Ambiente isolado Python para gerenciamento de dependências
- **Command Line Interface (CLI)**: Interface de linha de comando personalizada para execução de todos os modos do projeto

### 📁 **Estrutura do Projeto**
- **image_processing.py**: Módulo core para processamento de imagens e extração de features
- **ml_pipeline.py**: Pipeline completo de machine learning com suporte a múltiplos algoritmos
- **demo.py**: Interface CLI para todos os modos de operação
- **dataset_inspector.py**: Utilitário para validação e inspeção de datasets
- **experiments.ipynb**: Notebook interativo para experimentos e demonstrações
- **start.bat**: Script de instalação automatizada para Windows

### 🎯 **Recursos Técnicos Implementados**
- **Suporte a Imagens Coloridas**: Processamento RGB completo vs escala de cinza
- **Features Avançadas**: Combinação de cor, textura e forma (HOG, LBP, momentos de Hu)
- **Modelos ML**: RandomForest tradicional + CNNs 1D/2D
- **Data Augmentation**: Rotação, flip, zoom, shift para aumentar datasets
- **Otimização**: Grid Search para hiperparâmetros de CNNs
- **Validação Robusta**: Separação train/test, métricas completas, matriz de confusão
- **Persistência de Modelos**: Salvamento híbrido para modelos CNN (HDF5 + joblib)
- **Tratamento de Erros**: Validação de arquivos corrompidos, tratamento de exceções

Este projeto representa uma implementação completa de visão computacional moderna, combinando técnicas tradicionais e deep learning para classificação de imagens com alto desempenho e facilidade de uso.

### 6. Inspecionar o dataset organizado em subpastas
```bash
python dataset_inspector.py --dataset dataset
```

## 🚀 Publicação no GitHub

Para publicar este projeto no GitHub:

1. **Limpe o projeto**:
   ```bash
   clean_for_github.bat
   ```

2. **Crie o repositório** no GitHub

3. **Clone e copie os arquivos**:
   ```bash
   git clone <seu-repo>
   cd <nome-do-repo>
   # Copie todos os arquivos do projeto
   ```

4. **Adicione ao Git**:
   ```bash
   git add .
   git commit -m "Initial commit: Projeto HARDCORE de Processamento de Imagens"
   git push origin main
   ```

### 📝 Arquivos Incluídos no GitHub
- ✅ Código fonte completo (Python)
- ✅ Scripts de instalação (`start.bat`)
- ✅ Scripts de limpeza (`clean_for_github.bat`)
- ✅ Documentação completa (`README.md`, `PROJECT_DESCRIPTION.md`)
- ✅ Notebook de experimentos (`experiments.ipynb`)
- ✅ Lista de dependências (`requirements.txt`)
- ✅ Arquivo `.gitignore` configurado

### 🚫 Arquivos Excluídos (não subir)
- ❌ Ambiente virtual (`.venv/`)
- ❌ Modelos treinados (`*.joblib`, `*.h5`)
- ❌ Resultados de testes (`output/`)
- ❌ Datasets grandes (`dataset/`)
- ❌ Cache Python (`__pycache__/`)
- ❌ Arquivos temporários
- Pipeline de ML com treinamento, avaliação e serialização de modelos.
- **NOVO**: Balanceamento de dataset com RandomOverSampler para classes minoritárias.
- **NOVO**: Suporte a Redes Neurais Convolucionais (CNN) como alternativa ao RandomForest.
- Validação do modelo com métricas de classificação (precisão, recall, F1-score).
- Conjunto de teste separado para avaliação imparcial.
- Inferência em tempo real com modelos treinados.
- Limpeza automática de imagens corrompidas no dataset.

## Como usar balanceamento e CNN

### Treinar com balanceamento
```bash
python demo.py --mode train --dataset dataset --output output --model-out model_balanced.joblib --balance
```

### Treinar com CNN
```bash
python demo.py --mode train --dataset dataset --output output --model-out model_cnn.joblib --model-type cnn
```

### Fazer predição
```bash
python demo.py --mode predict --input imagem.jpg --model-in model_cnn.joblib --output output
```

## Notebook de experimentos
Abra `experiments.ipynb` para:
- visualizar imagens e resultados dos filtros;
- analisar histogramas de cor e textura;
- construir um dataset de features;
- treinar e avaliar um classificador.

## Próximos passos

1. Expandir para detecção de objetos usando CNNs.
2. Integrar segmentação avançada (Watershed, U-Net).
3. Criar uma interface web ou desktop para exploração interativa.
