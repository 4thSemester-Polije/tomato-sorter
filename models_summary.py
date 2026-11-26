
import csv
import re
from pathlib import Path

from ml.resnext import create_model


def print_table(headers, rows):
    """Print tabel dengan lebar kolom otomatis."""
    rows = [[str(value) for value in row] for row in rows]

    widths = [
        max(
            len(str(header)),
            max((len(row[i]) for row in rows), default=0)
        )
        for i, header in enumerate(headers)
    ]

    fmt = " | ".join(f"{{:<{w}}}" for w in widths)

    print(fmt.format(*headers))
    print("-+-".join("-" * w for w in widths))

    for row in rows:
        print(fmt.format(*row))


def shape_text(shape):
    return str(tuple(shape))


def write_csv(filename, headers, rows):
    """Write one report table beside this script for easy opening in Excel."""
    path = Path(__file__).resolve().with_name(filename)
    with path.open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.writer(output)
        writer.writerow(headers)
        writer.writerows(rows)
    return path


def get_source_layers(layer):
    """Mengambil layer yang terhubung langsung ke input layer."""
    try:
        tensors = layer.input
        if not isinstance(tensors, (list, tuple)):
            tensors = [tensors]

        sources = get_source_layer_objects(tensors)
        return ", ".join(dict.fromkeys(source.name for source in sources)) or "-"

    except (AttributeError, ValueError):
        return "-"


def get_source_layer_objects(tensors):
    """Return unique source layer objects for the given input tensors."""
    sources = []
    for tensor in tensors:
        history = getattr(tensor, "_keras_history", None)
        if history is None:
            continue

        source = (
            getattr(history, "operation", None)
            or getattr(history, "layer", None)
        )
        if source is None:
            try:
                source = history[0]
            except (TypeError, IndexError):
                continue

        if all(source is not existing for existing in sources):
            sources.append(source)
    return sources


def layer_output(layer):
    try:
        outputs = layer.output

        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]

        return ", ".join(shape_text(x.shape) for x in outputs)

    except (AttributeError, ValueError):
        return "-"


def layer_operation(layer):
    """Mengambil informasi operasi yang penting saja."""
    config = layer.get_config()
    layer_type = type(layer).__name__

    details = []

    for key in (
        "filters",
        "kernel_size",
        "strides",
        "groups",
        "pool_size",
        "padding",
        "activation",
        "axis",
        "units",
        "rate",
    ):
        if key in config:
            details.append(f"{key}={config[key]}")

    if details:
        return f"{layer_type} ({', '.join(details)})"

    return layer_type


def grouped_convolutions(backbone):
    """Map each library-expanded grouped convolution to one logical operation."""
    grouped = {}
    for layer in backbone.layers:
        if type(layer).__name__ != "Concatenate":
            continue

        try:
            inputs = layer.input
            if not isinstance(inputs, (list, tuple)):
                inputs = [inputs]
        except (AttributeError, ValueError):
            continue

        conv_layers = get_source_layer_objects(inputs)
        if len(conv_layers) < 2 or any(
            type(source).__name__ != "Conv2D" for source in conv_layers
        ):
            continue

        slice_layers = []
        valid_group = True
        for conv_layer in conv_layers:
            try:
                conv_inputs = conv_layer.input
                if not isinstance(conv_inputs, (list, tuple)):
                    conv_inputs = [conv_inputs]
            except (AttributeError, ValueError):
                valid_group = False
                break

            conv_sources = get_source_layer_objects(conv_inputs)
            if len(conv_sources) != 1 or type(conv_sources[0]).__name__ != "Lambda":
                valid_group = False
                break
            slice_layers.append(conv_sources[0])

        if not valid_group:
            continue

        source_layers = []
        for slice_layer in slice_layers:
            try:
                slice_inputs = slice_layer.input
                if not isinstance(slice_inputs, (list, tuple)):
                    slice_inputs = [slice_inputs]
            except (AttributeError, ValueError):
                continue
            for source in get_source_layer_objects(slice_inputs):
                if all(source is not existing for existing in source_layers):
                    source_layers.append(source)
        if len(source_layers) != 1:
            continue

        first_config = conv_layers[0].get_config()
        grouped_name = f"grouped_conv/{layer.name}"
        descriptor = {
            "name": grouped_name,
            "concat": layer,
            "convs": conv_layers,
            "source": source_layers[0],
            "groups": len(conv_layers),
            "filters": sum(
                item.get_config().get("filters", 0) for item in conv_layers
            ),
            "kernel_size": first_config.get("kernel_size"),
            "strides": first_config.get("strides"),
            "stage": "",
            "block": "",
            "operation": (
                f"GroupedConv2D (groups={len(conv_layers)}, "
                f"filters={sum(item.get_config().get('filters', 0) for item in conv_layers)}, "
                f"kernel_size={first_config.get('kernel_size')}, "
                f"strides={first_config.get('strides')})"
            ),
            "parameters": sum(item.count_params() for item in conv_layers),
            "trainable": any(item.trainable for item in conv_layers),
        }
        grouped[layer.name] = descriptor
        for conv_layer in conv_layers:
            grouped[conv_layer.name] = descriptor

    return grouped


# ============================================================
# 1. CREATE MODEL
# ============================================================

model, backbone, preprocess_input = create_model(
    task="ripeness",
    num_classes=3,
    architecture="resnext101"
)


# ============================================================
# 2. MODEL OVERVIEW
# ============================================================

total_params = model.count_params()

trainable_params = sum(
    int(w.shape.num_elements())
    for w in model.trainable_weights
)

non_trainable_params = sum(
    int(w.shape.num_elements())
    for w in model.non_trainable_weights
)

print("\n" + "=" * 100)
print("RESNEXT-101 MODEL ARCHITECTURE REPORT")
print("=" * 100)

overview = [
    ("Architecture", "ResNeXt-101"),
    ("Task", "Tomato Ripeness Classification"),
    ("Number of classes", 3),
    ("Model input", shape_text(model.input.shape)),
    ("Model output", shape_text(model.output.shape)),
    ("Backbone", "ResNeXt-101"),
    ("Outer Keras layers", len(model.layers)),
    ("Layers inside backbone", len(backbone.layers)),
    ("Expanded Keras layer count", len(model.layers) + len(backbone.layers) - 1),
    ("Backbone parameters", f"{backbone.count_params():,}"),
    ("Total parameters", f"{total_params:,}"),
    ("Trainable parameters", f"{trainable_params:,}"),
    ("Non-trainable parameters", f"{non_trainable_params:,}"),
]

print_table(("Information", "Value"), overview)


# ============================================================
# 3. FULL MODEL LAYERS (LOGICAL VIEW)
# ============================================================

print("\n" + "=" * 100)
print("FULL MODEL LAYER FLOW - INPUT TO CLASSIFICATION OUTPUT")
print("=" * 100)
print(
    "Backbone adalah sebuah model di dalam model utama. Di tabel ini isinya "
    "diurai menjadi layer internal; baris Lambda dan pecahan grouped conv "
    "digabung agar alur arsitektur lebih mudah dibaca."
)

grouped = grouped_convolutions(backbone)
grouped_rows_added = set()
backbone_aliases = {}
detail_rows = []
block_pattern = re.compile(r"^stage(\d+)_unit(\d+)_")

for stage_bn2 in backbone.layers:
    match = re.match(r"^stage(\d+)_unit(\d+)_bn2$", stage_bn2.name)
    if match is None:
        continue
    try:
        tensors = stage_bn2.input
        if not isinstance(tensors, (list, tuple)):
            tensors = [tensors]
    except (AttributeError, ValueError):
        continue
    for source in get_source_layer_objects(tensors):
        descriptor = grouped.get(source.name)
        if descriptor is not None:
            descriptor["stage"] = int(match.group(1))
            descriptor["block"] = int(match.group(2))
            break


def append_detail_row(
    name, connected, operation, output, parameters, trainable, stage="", block=""
):
    detail_rows.append((
        len(detail_rows) + 1,
        stage,
        block,
        name,
        connected or "-",
        operation,
        output,
        parameters,
        "yes" if trainable else "no",
    ))


for layer in model.layers:
    if layer is not backbone:
        try:
            tensors = layer.input
            if not isinstance(tensors, (list, tuple)):
                tensors = [tensors]
            source_layers = get_source_layer_objects(tensors)
        except (AttributeError, ValueError):
            source_layers = []

        source_names = []
        for source in source_layers:
            if source is backbone:
                source_names.append(
                    f"backbone/{backbone.layers[-1].name}"
                )
            else:
                source_names.append(source.name)
        append_detail_row(
            layer.name,
            ", ".join(dict.fromkeys(source_names)) or "-",
            layer_operation(layer),
            layer_output(layer),
            layer.count_params(),
            layer.trainable,
        )
        continue

    for backbone_layer in backbone.layers:
        if type(backbone_layer).__name__ == "Lambda":
            continue

        group = grouped.get(backbone_layer.name)
        if group is not None:
            group_key = group["name"]
            if group_key in grouped_rows_added:
                continue
            grouped_rows_added.add(group_key)
            backbone_aliases[group["concat"].name] = group_key
            source = group["source"]
            if type(source).__name__ == "InputLayer":
                connected = "random_rotation"
            else:
                connected = backbone_aliases.get(
                    source.name, f"backbone/{source.name}"
                )
            append_detail_row(
                f"backbone/{group_key}",
                connected,
                group["operation"],
                layer_output(group["concat"]),
                group["parameters"],
                group["trainable"],
                group["stage"],
                group["block"],
            )
            continue

        if (
            type(backbone_layer).__name__ == "Concatenate"
            and backbone_layer.name in grouped
        ):
            # The grouped convolution's concatenate is represented by its
            # single logical GroupedConv2D row above.
            continue

        connected_names = []
        if type(backbone_layer).__name__ == "InputLayer":
            connected_names.append("random_rotation")
        else:
            try:
                tensors = backbone_layer.input
                if not isinstance(tensors, (list, tuple)):
                    tensors = [tensors]
                sources = get_source_layer_objects(tensors)
            except (AttributeError, ValueError):
                sources = []

            for source in sources:
                if type(source).__name__ == "InputLayer":
                    connected_names.append("random_rotation")
                else:
                    connected_names.append(
                        backbone_aliases.get(
                            source.name, f"backbone/{source.name}"
                        )
                    )

        append_detail_row(
            f"backbone/{backbone_layer.name}",
            ", ".join(dict.fromkeys(connected_names)),
            layer_operation(backbone_layer),
            layer_output(backbone_layer),
            backbone_layer.count_params(),
            backbone_layer.trainable,
            int(match.group(1)) if (match := block_pattern.match(backbone_layer.name)) else "",
            int(match.group(2)) if match else "",
        )

    # Map the nested backbone model's output to its final internal layer for
    # the connection shown on the first classification-head layer.
    backbone_aliases[backbone.name] = f"backbone/{backbone.layers[-1].name}"

logical_layer_csv = write_csv(
    "resnext101_logical_layers.csv",
    (
        "No.",
        "Stage",
        "Bottleneck Block",
        "Layer",
        "Connected From",
        "Operation / Details",
        "Output Shape",
        "Parameters",
        "Trainable",
    ),
    detail_rows,
)

print(f"\nLogical layer rows written: {len(detail_rows)}")
print(f"Logical layer CSV: {logical_layer_csv}")
print(
    "Actual expanded Keras layers (including Lambda and grouped-convolution "
    f"implementation layers): {len(model.layers) + len(backbone.layers) - 1}"
)

conv7 = next((layer for layer in backbone.layers if layer.name == "conv0"), None)
print("\nStem Conv 7x7 verification:")
if conv7 is None:
    print("Layer 'conv0' was not found in the constructed backbone.")
else:
    conv7_config = conv7.get_config()
    kernel_size = tuple(conv7_config.get("kernel_size", ()))
    strides = tuple(conv7_config.get("strides", ()))
    print(
        f"{conv7.name}: kernel_size={kernel_size}, strides={strides}, "
        f"filters={conv7_config.get('filters')}"
    )
    print(
        "Conv 7x7 stride 2 is used: "
        f"{kernel_size == (7, 7) and strides == (2, 2)}"
    )


# ============================================================
# 5. RESNEXT STAGE SUMMARY
# ============================================================

print("\n" + "=" * 100)
print("RESNEXT-101 STAGE SUMMARY")
print("=" * 100)

block_details = []
block_start_pattern = re.compile(r"^stage(\d+)_unit(\d+)_conv1$")

for conv1 in backbone.layers:
    match = block_start_pattern.match(conv1.name)
    if match is None:
        continue

    stage_number = int(match.group(1))
    block_number = int(match.group(2))
    block_prefix = f"stage{stage_number}_unit{block_number}_"
    bn2 = next(
        (
            layer for layer in backbone.layers
            if layer.name == f"{block_prefix}bn2"
        ),
        None,
    )
    block_output = next(
        (
            layer for layer in backbone.layers
            if layer.name == f"{block_prefix}relu"
        ),
        None,
    )
    if bn2 is None or block_output is None:
        continue

    group = None
    try:
        tensors = bn2.input
        if not isinstance(tensors, (list, tuple)):
            tensors = [tensors]
        for source in get_source_layer_objects(tensors):
            group = grouped.get(source.name)
            if group is not None:
                break
    except (AttributeError, ValueError):
        pass

    shortcut = (
        "projection (1x1 conv + BN)"
        if f"{block_prefix}sc" in {layer.name for layer in backbone.layers}
        else "identity"
    )
    named_block_parameters = sum(
        layer.count_params()
        for layer in backbone.layers
        if layer.name.startswith(block_prefix)
    )
    grouped_parameters = group["parameters"] if group else 0

    block_details.append({
        "stage": stage_number,
        "block": block_number,
        "block_type": "projection" if "projection" in shortcut else "identity",
        "input_shape": shape_text(conv1.input.shape),
        "output_shape": shape_text(block_output.output.shape),
        "bottleneck_filters": conv1.get_config().get("filters", ""),
        "group_count": group["groups"] if group else "",
        "grouped_conv_filters": group["filters"] if group else "",
        "grouped_conv_kernel": group["kernel_size"] if group else "",
        "grouped_conv_stride": group["strides"] if group else "",
        "shortcut": shortcut,
        "parameters": named_block_parameters + grouped_parameters,
    })

block_headers = (
    "Stage",
    "Bottleneck Block",
    "Shortcut Type",
    "Input Shape",
    "Output Shape",
    "Bottleneck 1x1 Filters",
    "Grouped Conv Groups",
    "Grouped Conv Filters",
    "Grouped Conv Kernel",
    "Grouped Conv Stride",
    "Shortcut Details",
    "Parameters",
)
block_rows = [
    (
        row["stage"],
        row["block"],
        row["block_type"],
        row["input_shape"],
        row["output_shape"],
        row["bottleneck_filters"],
        row["group_count"],
        row["grouped_conv_filters"],
        row["grouped_conv_kernel"],
        row["grouped_conv_stride"],
        row["shortcut"],
        row["parameters"],
    )
    for row in block_details
]
block_csv = write_csv(
    "resnext101_bottleneck_blocks.csv",
    block_headers,
    block_rows,
)

stage_rows = []
for stage_number in sorted({row["stage"] for row in block_details}):
    stage_blocks = [
        row for row in block_details if row["stage"] == stage_number
    ]
    first_block = stage_blocks[0]
    last_block = stage_blocks[-1]
    stage_rows.append((
        f"Stage {stage_number}",
        len(stage_blocks),
        f"unit{first_block['block']}",
        f"unit{last_block['block']}",
        first_block["input_shape"],
        last_block["output_shape"],
        "32 per bottleneck" if all(
            row["group_count"] == 32 for row in stage_blocks
        ) else "see block CSV",
    ))

stage_headers = (
    "Stage",
    "Bottleneck Blocks",
    "First Block",
    "Last Block",
    "Input Shape",
    "Output Shape",
    "Grouped Conv",
)
stage_csv = write_csv(
    "resnext101_stage_summary.csv",
    stage_headers,
    stage_rows,
)

print_table(
    (
        "Stage",
        "Bottleneck Blocks",
        "First Block",
        "Last Block",
        "Input Shape",
        "Output Shape",
        "Grouped Conv",
    ),
    stage_rows
)
print(f"\nBottleneck block CSV: {block_csv}")
print(f"Stage summary CSV: {stage_csv}")


# ============================================================
# 6. PARAMETER SUMMARY
# ============================================================

print("\n" + "=" * 100)
print("PARAMETER SUMMARY")
print("=" * 100)

parameter_rows = [
    ("Backbone", f"{backbone.count_params():,}"),
    ("Total Parameters", f"{total_params:,}"),
    ("Trainable Parameters", f"{trainable_params:,}"),
    ("Non-trainable Parameters", f"{non_trainable_params:,}"),
    (
        "Trainable Percentage",
        f"{trainable_params / total_params * 100:.2f}%"
        if total_params else "0%"
    ),
]

print_table(("Parameter Type", "Count"), parameter_rows)

print("\n" + "=" * 100)
print("ARCHITECTURE REPORT COMPLETED")
print("=" * 100)
