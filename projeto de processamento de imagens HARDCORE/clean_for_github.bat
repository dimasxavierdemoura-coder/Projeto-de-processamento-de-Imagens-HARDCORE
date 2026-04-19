@echo off
echo ================================================
echo    LIMPEZA DO PROJETO PARA GITHUB
echo ================================================
echo.
echo Este script vai limpar arquivos temporarios
echo antes de subir o projeto pro GitHub.
echo.
echo Pressione qualquer tecla para continuar...
pause >nul

echo.
echo Removendo ambiente virtual...
if exist ".venv" (
    rmdir /s /q ".venv"
    echo Ambiente virtual removido.
) else (
    echo Ambiente virtual nao encontrado.
)

echo.
echo Removendo modelos treinados...
del /q "*.joblib" 2>nul
del /q "*.h5" 2>nul
echo Modelos removidos.

echo.
echo Removendo pasta output...
if exist "output" (
    rmdir /s /q "output"
    echo Pasta output removida.
) else (
    echo Pasta output nao encontrada.
)

echo.
echo Removendo pasta dataset...
if exist "dataset" (
    rmdir /s /q "dataset"
    echo Pasta dataset removida.
) else (
    echo Pasta dataset nao encontrada.
)

echo.
echo Removendo pasta __pycache__...
if exist "__pycache__" (
    rmdir /s /q "__pycache__"
    echo Pasta __pycache__ removida.
) else (
    echo Pasta __pycache__ nao encontrada.
)

echo.
echo ================================================
echo    LIMPEZA CONCLUIDA!
echo ================================================
echo.
echo Arquivos removidos:
echo - .venv/ (ambiente virtual)
echo - *.joblib (modelos treinados)
echo - *.h5 (modelos TensorFlow)
echo - output/ (resultados de testes)
echo - dataset/ (dados de exemplo)
echo - __pycache__/ (bytecode Python)
echo.
echo O projeto esta pronto para GitHub!
echo.
echo Pressione qualquer tecla para sair...
pause >nul