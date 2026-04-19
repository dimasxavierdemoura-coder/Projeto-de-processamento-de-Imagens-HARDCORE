@echo off
echo ================================================
echo    PROCESSAMENTO DE IMAGENS - PROJETO HARDCORE
echo ================================================
echo.
echo ATENCAO: Antes de continuar, por favor abasteca
echo a pasta "dataset" com suas imagens organizadas
echo em subpastas por classe (ex: dataset/frutas, dataset/animais).
echo.
echo Pressione qualquer tecla para continuar...
pause >nul

echo.
echo Verificando instalacao do Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERRO: Python nao encontrado. Instale Python 3.10+ primeiro.
    echo Baixe em: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo Python encontrado. Criando ambiente virtual...
python -m venv .venv
if errorlevel 1 (
    echo ERRO: Falha ao criar ambiente virtual.
    pause
    exit /b 1
)

echo Ativando ambiente virtual...
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERRO: Falha ao ativar ambiente virtual.
    pause
    exit /b 1
)

echo Instalando dependencias...
pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo ERRO: Falha ao instalar dependencias.
    pause
    exit /b 1
)

echo.
echo ================================================
echo    INSTALACAO CONCLUIDA COM SUCESSO!
echo ================================================
echo.
echo Verificando dataset...
if exist "dataset" (
    echo Dataset encontrado. Executando inspecao...
    python dataset_inspector.py
    echo.
    echo Deseja executar um treinamento de exemplo? (s/n)
    set /p choice=
    if /i "!choice!"=="s" (
        echo Executando treinamento RandomForest...
        python demo.py --mode train --dataset dataset --output output --model-out model_rf.joblib --model-type rf
        echo.
        echo Treinamento concluido! Modelo salvo como model_rf.joblib
    )
) else (
    echo Pasta dataset nao encontrada. Crie a pasta dataset e adicione suas imagens organizadas por classe.
)

echo.
echo Agora voce pode usar os comandos:
echo.
echo - Processar bordas: python demo.py --mode edges --input imagem.jpg --output output
echo - Treinar modelo: python demo.py --mode train --dataset dataset --output output --model-out model.joblib
echo - Benchmark RF vs CNN: python demo.py --mode benchmark --dataset dataset --output output --balance --augment
echo - Grid search CNN: python demo.py --mode grid-search --dataset dataset --output output --balance --augment
echo - Predicao: python demo.py --mode predict --input imagem.jpg --model-in model.joblib --output output
echo.
echo Para mais detalhes, consulte o README.md
echo.
echo Pressione qualquer tecla para sair...
pause >nul