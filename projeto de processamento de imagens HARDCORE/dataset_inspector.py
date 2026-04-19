import argparse
import os

VALID_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "tiff", "tif"}


def is_image_file(filename):
    return filename.lower().split(".")[-1] in VALID_EXTENSIONS


def inspect_dataset(dataset_dir):
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Diretório não encontrado: {dataset_dir}")

    structure = {}
    problems = []

    for root, dirs, files in os.walk(dataset_dir):
        relative_root = os.path.relpath(root, dataset_dir)
        if relative_root == ".":
            continue

        class_name = os.path.normpath(relative_root).split(os.sep)[0]
        for file in files:
            if not is_image_file(file):
                problems.append(os.path.join(root, file))

        class_files = [f for f in files if is_image_file(f)]
        structure.setdefault(class_name, []).extend(class_files)

    return structure, problems


def print_structure(structure):
    print("Dataset structure:")
    for class_name, files in sorted(structure.items()):
        print(f"- {class_name}: {len(files)} imagens")


def print_problems(problems):
    if not problems:
        print("Nenhum problema encontrado nos arquivos de imagem.")
        return

    print("Arquivos não válidos encontrados:")
    for path in problems:
        print(f"- {path}")


def main():
    parser = argparse.ArgumentParser(description="Inspecionar e validar a estrutura do dataset")
    parser.add_argument("--dataset", default="dataset", help="Caminho para o diretório dataset")
    args = parser.parse_args()

    structure, problems = inspect_dataset(args.dataset)
    print_structure(structure)
    print()
    print_problems(problems)

    if structure:
        print()
        print("Estrutura de classes detectada com sucesso.")
    else:
        print("Nenhuma classe detectada. Verifique se o dataset contém subpastas com imagens.")


if __name__ == "__main__":
    main()
