import argparse
import json
import os

from image_processing import (
    adaptive_threshold,
    apply_gaussian_blur,
    canny_edge,
    closing,
    color_histogram_features,
    convert_to_gray,
    dilate,
    erode,
    extract_image_features,
    hit_or_miss,
    load_image,
    load_grayscale,
    opening,
    prewitt_edge,
    save_image,
    sobel_edge,
    laplacian_edge,
    threshold_binary,
    get_default_hit_or_miss_kernel,
)
from ml_pipeline import (
    load_dataset_features, 
    load_dataset_images,
    save_model, 
    train_classifier, 
    load_model, 
    predict_image,
    grid_search_cnn,
    benchmark_models
)


def ensure_output_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def save_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def run_edge_pipeline(image_path, output_dir):
    image = load_image(image_path)
    save_image(os.path.join(output_dir, "original.png"), image)
    save_image(os.path.join(output_dir, "sobel.png"), sobel_edge(image))
    save_image(os.path.join(output_dir, "prewitt.png"), prewitt_edge(image))
    save_image(os.path.join(output_dir, "laplacian.png"), laplacian_edge(image))
    save_image(os.path.join(output_dir, "canny.png"), canny_edge(image))
    print("Bordas geradas com sucesso.")


def run_morphology_pipeline(image_path, output_dir, thresh):
    gray = load_grayscale(image_path)
    binary = threshold_binary(gray, threshold=thresh)
    save_image(os.path.join(output_dir, "binary.png"), binary)
    save_image(os.path.join(output_dir, "dilate.png"), dilate(binary, kernel_size=5))
    save_image(os.path.join(output_dir, "erode.png"), erode(binary, kernel_size=5))
    save_image(os.path.join(output_dir, "opening.png"), opening(binary, kernel_size=5))
    save_image(os.path.join(output_dir, "closing.png"), closing(binary, kernel_size=5))
    save_image(os.path.join(output_dir, "adaptive_binary.png"), adaptive_threshold(gray, block_size=15, c=4))
    save_image(
        os.path.join(output_dir, "hit_or_miss.png"),
        hit_or_miss(binary, get_default_hit_or_miss_kernel()),
    )
    print("Operações morfológicas concluídas.")


def run_feature_extraction(image_path, output_dir):
    image = load_image(image_path)
    features = extract_image_features(image)
    features["color_hist_mean"] = float(sum(color_histogram_features(image).values()) / len(color_histogram_features(image)))
    save_json(os.path.join(output_dir, "features.json"), features)
    print("Extração de features concluída.")


def run_training(dataset_dir, output_dir, model_out, balance=False, model_type="rf", augment=False, grid_search=False):
    if model_type in ["cnn_2d"] or grid_search:
        # Load images for CNN
        X, y = load_dataset_images(dataset_dir, target_size=(128, 128), balance=balance, augment=augment)
    else:
        # Load features for RF/CNN_1D
        X, y = load_dataset_features(dataset_dir, balance=balance)
    
    if grid_search and model_type == "cnn_2d":
        print("Executando grid search para CNN 2D...")
        param_grid = [
            {'filters1': 32, 'filters2': 64, 'dense_units': 128, 'dropout_rate': 0.3, 'learning_rate': 0.001},
            {'filters1': 64, 'filters2': 128, 'dense_units': 256, 'dropout_rate': 0.4, 'learning_rate': 0.001},
            {'filters1': 32, 'filters2': 64, 'dense_units': 128, 'dropout_rate': 0.5, 'learning_rate': 0.0001},
        ]
        model, best_params, best_score = grid_search_cnn(X, y, param_grid)
        print(f"Melhores parâmetros: {best_params}")
    else:
        model, report = train_classifier(X, y, model_type=model_type)
    
    save_model(model, model_out)
    save_json(os.path.join(output_dir, "training_report.json"), report)
    print(f"Treinamento concluído. Modelo salvo em: {model_out}")


def run_grid_search(dataset_dir, output_dir, balance=False, augment=False):
    print("=== GRID SEARCH CNN ===")
    X_images, y_images = load_dataset_images(dataset_dir, balance=balance, augment=augment)
    best_params, best_score = grid_search_cnn(X_images, y_images)
    results = {
        'best_params': best_params,
        'best_score': best_score
    }
    save_json(os.path.join(output_dir, "grid_search_results.json"), results)
    print(f"Melhores parâmetros: {best_params}")
    print(".4f")
    print("Grid search concluído. Resultados salvos.")
    results = benchmark_models(dataset_dir)
    save_json(os.path.join(output_dir, "benchmark_results.json"), results)
    print("Benchmark concluído. Resultados salvos.")


def run_predict(image_path, model_path, output_dir):
    model = load_model(model_path)
    prediction = predict_image(model, image_path)
    result = {"prediction": str(prediction)}  # Convert to string
    save_json(os.path.join(output_dir, "prediction.json"), result)
    print(f"Predição: {prediction}")


def main():
    parser = argparse.ArgumentParser(description="Demo hardcore de processamento de imagens")
    parser.add_argument("--mode", required=True, choices=["edges", "morphology", "features", "train", "predict", "benchmark", "grid-search"], help="Modo de execução")
    parser.add_argument("--input", help="Caminho para a imagem de entrada")
    parser.add_argument("--dataset", help="Diretório de dataset rotulado para treinamento")
    parser.add_argument("--output", required=True, help="Diretório de saída")
    parser.add_argument("--model-out", default="model.joblib", help="Caminho para salvar o modelo treinado")
    parser.add_argument("--model-in", default="model.joblib", help="Caminho para o modelo treinado para predição")
    parser.add_argument("--thresh", type=int, default=128, help="Limiar para binarização")
    parser.add_argument("--balance", action="store_true", help="Aplicar balanceamento ao dataset")
    parser.add_argument("--model-type", choices=["rf", "cnn_1d", "cnn_2d"], default="rf", help="Tipo de modelo: rf (RandomForest), cnn_1d (CNN em features), cnn_2d (CNN em imagens)")
    parser.add_argument("--augment", action="store_true", help="Aplicar data augmentation ao dataset")
    parser.add_argument("--grid-search", action="store_true", help="Executar grid search para otimização de hiperparâmetros")
    args = parser.parse_args()

    output_dir = ensure_output_dir(args.output)

    if args.mode in {"edges", "morphology", "features", "predict"} and not args.input:
        raise ValueError("--input é obrigatório para modes edges, morphology, features e predict")

    if args.mode == "edges":
        run_edge_pipeline(args.input, output_dir)
    elif args.mode == "morphology":
        run_morphology_pipeline(args.input, output_dir, args.thresh)
    elif args.mode == "features":
        run_feature_extraction(args.input, output_dir)
    elif args.mode == "train":
        if not args.dataset:
            raise ValueError("--dataset é obrigatório para o modo train")
        run_training(args.dataset, output_dir, args.model_out, 
                    balance=args.balance, model_type=args.model_type,
                    augment=args.augment, grid_search=args.grid_search)
    elif args.mode == "predict":
        run_predict(args.input, args.model_in, output_dir)
    elif args.mode == "benchmark":
        if not args.dataset:
            raise ValueError("--dataset é obrigatório para o modo benchmark")
        run_benchmark(args.dataset, output_dir)
    elif args.mode == "grid-search":
        if not args.dataset:
            raise ValueError("--dataset é obrigatório para o modo grid-search")
        run_grid_search(args.dataset, output_dir, balance=args.balance, augment=args.augment)

    print(f"Modo {args.mode} finalizado. Resultados em: {output_dir}")


if __name__ == "__main__":
    main()
