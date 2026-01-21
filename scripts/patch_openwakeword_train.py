from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _patch_train_py(train_py: Path) -> bool:
    original = _load_text(train_py)
    normalized = original.replace("\r\n", "\n")
    changed = False

    old_val_block = (
        "        X_val_fp = np.load(config[\"false_positive_validation_data_path\"])\n"
        "        X_val_fp = np.array([X_val_fp[i:i+input_shape[0]] for i in range(0, X_val_fp.shape[0]-input_shape[0], 1)])  # reshape to match model\n"
        "        X_val_fp_labels = np.zeros(X_val_fp.shape[0]).astype(np.float32)\n"
        "        X_val_fp = torch.utils.data.DataLoader(\n"
        "            torch.utils.data.TensorDataset(torch.from_numpy(X_val_fp), torch.from_numpy(X_val_fp_labels)),\n"
        "            batch_size=len(X_val_fp_labels)\n"
        "        )\n"
        "\n"
    )

    new_val_block = (
        "        fp_path = config.get(\"false_positive_validation_data_path\")\n"
        "        if fp_path:\n"
        "            X_val_fp = np.load(fp_path)\n"
        "            X_val_fp = np.array([X_val_fp[i:i+input_shape[0]] for i in range(0, X_val_fp.shape[0]-input_shape[0], 1)])  # reshape to match model\n"
        "            X_val_fp_labels = np.zeros(X_val_fp.shape[0]).astype(np.float32)\n"
        "            X_val_fp = torch.utils.data.DataLoader(\n"
        "                torch.utils.data.TensorDataset(torch.from_numpy(X_val_fp), torch.from_numpy(X_val_fp_labels)),\n"
        "                batch_size=len(X_val_fp_labels)\n"
        "            )\n"
        "        else:\n"
        "            X_val_fp = None\n"
        "\n"
    )

    old_train_block = (
        "        # Run auto training\n"
        "        best_model = oww.auto_train(\n"
        "            X_train=X_train,\n"
        "            X_val=X_val,\n"
        "            false_positive_val_data=X_val_fp,\n"
        "            steps=config[\"steps\"],\n"
        "            max_negative_weight=config[\"max_negative_weight\"],\n"
        "            target_fp_per_hour=config[\"target_false_positives_per_hour\"],\n"
        "        )\n"
        "\n"
    )

    new_train_block = (
        "        # Run auto training\n"
        "        auto_train_kwargs = dict(\n"
        "            X_train=X_train,\n"
        "            X_val=X_val,\n"
        "            steps=config[\"steps\"],\n"
        "            max_negative_weight=config[\"max_negative_weight\"],\n"
        "            target_fp_per_hour=config[\"target_false_positives_per_hour\"],\n"
        "        )\n"
        "        if X_val_fp is not None:\n"
        "            auto_train_kwargs[\"false_positive_val_data\"] = X_val_fp\n"
        "\n"
        "        best_model = oww.auto_train(**auto_train_kwargs)\n"
        "\n"
    )

    if new_val_block in normalized and new_train_block in normalized:
        pass
    else:
        if old_val_block not in normalized or old_train_block not in normalized:
            raise ValueError(
                "Не удалось найти ожидаемые блоки в openwakeword/train.py. "
                "Проверьте версию пакета и обновите патч при необходимости."
            )

        normalized = normalized.replace(old_val_block, new_val_block)
        normalized = normalized.replace(old_train_block, new_train_block)
        changed = True

    old_generate_import = "    from generate_samples import generate_samples\n"
    new_generate_import = (
        "    from generate_samples import generate_samples\n"
        "    import inspect\n"
        "\n"
        "    def _call_generate_samples(**kwargs):\n"
        "        sig = inspect.signature(generate_samples)\n"
        "        if \"model\" in sig.parameters:\n"
        "            model_value = kwargs.get(\"model\") or config.get(\"piper_model_path\")\n"
        "            if model_value:\n"
        "                kwargs[\"model\"] = model_value\n"
        "            else:\n"
        "                raise ValueError(\n"
        "                    \"Missing piper_model_path in training_config.yaml for this version \"\n"
        "                    \"of piper-sample-generator.\"\n"
        "                )\n"
        "        else:\n"
        "            kwargs.pop(\"model\", None)\n"
        "        return generate_samples(**kwargs)\n"
        "\n"
    )

    if "_call_generate_samples" not in normalized:
        if old_generate_import not in normalized:
            raise ValueError(
                "Не удалось найти импорт generate_samples в openwakeword/train.py. "
                "Проверьте версию пакета и обновите патч при необходимости."
            )
        normalized = normalized.replace(old_generate_import, new_generate_import)
        normalized = normalized.replace("generate_samples(", "_call_generate_samples(")
        changed = True

    if changed:
        _write_text(train_py, normalized)
    return changed


def main() -> None:
    spec = importlib.util.find_spec("openwakeword")
    if not spec or not spec.submodule_search_locations:
        raise SystemExit("openwakeword не установлен. Установите пакет перед патчем.")

    package_dir = Path(list(spec.submodule_search_locations)[0])
    train_py = package_dir / "train.py"
    if not train_py.exists():
        raise SystemExit(f"Не найден файл {train_py}")

    try:
        changed = _patch_train_py(train_py)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if changed:
        print(f"Патч применен: {train_py}")
    else:
        print("Патч уже применен.")


if __name__ == "__main__":
    main()
