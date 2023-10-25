from io import BufferedReader
import struct
from pathlib import Path
import numpy as np


def load_gguf(file_path: Path) -> dict:
    gguf_file = {
        "header": {
            "magic": 0,
            "version": 0,
            "tensor_count": 0,
            "metadata_kv_count": 0,
            "metadata_kv": {},  # {key: List[value_type, value], ...}
        },
        "tensor_infos": {},
        "_padding": [],
        "tensor_data": [],
    }

    def parse_metadata_value(f: BufferedReader, value_t: int):
        if value_t == 0:  # GGUF_METADATA_VALUE_TYPE_UINT8
            return int.from_bytes(f.read(1), "little")
        elif value_t == 1:  # GGUF_METADATA_VALUE_TYPE_INT8
            return int.from_bytes(f.read(1), "little", signed=True)
        elif value_t == 2:  # GGUF_METADATA_VALUE_TYPE_UINT16
            return int.from_bytes(f.read(2), "little")
        elif value_t == 3:  # GGUF_METADATA_VALUE_TYPE_INT16
            return int.from_bytes(f.read(2), "little", signed=True)
        elif value_t == 4:  # GGUF_METADATA_VALUE_TYPE_UINT32
            return int.from_bytes(f.read(4), "little")
        elif value_t == 5:  # GGUF_METADATA_VALUE_TYPE_INT32
            return int.from_bytes(f.read(4), "little", signed=True)
        elif value_t == 6:  # GGUF_METADATA_VALUE_TYPE_FLOAT32
            return struct.unpack("<f", f.read(4))[0]
        elif value_t == 7:  # GGUF_METADATA_VALUE_TYPE_BOOL
            return bool(int.from_bytes(f.read(1), "little"))
        elif value_t == 8:  # GGUF_METADATA_VALUE_TYPE_STRING
            return f.read(int.from_bytes(f.read(8), "little")).decode("utf-8")
        elif value_t == 9:  # GGUF_METADATA_VALUE_TYPE_ARRAY
            val_kind = int.from_bytes(f.read(4), "little")
            return [
                (
                    parse_metadata_value(
                        f,
                        val_kind,
                    )
                )
                for _ in range(int.from_bytes(f.read(8), "little"))
            ]
        elif value_t == 10:  # GGUF_METADATA_VALUE_TYPE_UINT64
            return int.from_bytes(f.read(8), "little")
        elif value_t == 11:  # GGUF_METADATA_VALUE_TYPE_INT64
            return int.from_bytes(f.read(8), "little", signed=True)
        elif value_t == 12:  # GGUF_METADATA_VALUE_TYPE_FLOAT64
            return struct.unpack("<d", f.read(8))[0]
        else:
            raise ValueError(f"Invalid GGUF metadata value type: {value_t}")

    with open("weights/llama-2-7b.Q4_0.gguf", "rb") as f:
        gguf_file["header"]["magic"] = f.read(4).decode("utf-8")
        if gguf_file["header"]["magic"] != "GGUF":
            raise ValueError(
                f"Invalid GGUF file header magic: {gguf_file['header']['magic']}"
            )
        gguf_file["header"]["version"] = int.from_bytes(f.read(4), byteorder="little")
        gguf_file["header"]["tensor_count"] = int.from_bytes(f.read(8), "little")
        gguf_file["header"]["metadata_kv_count"] = int.from_bytes(f.read(8), "little")
        for _ in range(gguf_file["header"]["metadata_kv_count"]):
            key = f.read(int.from_bytes(f.read(8), "little")).decode("ascii")
            gguf_file["header"]["metadata_kv"][key] = [
                int.from_bytes(f.read(4), "little")
            ]
            gguf_file["header"]["metadata_kv"][key].append(
                parse_metadata_value(f, gguf_file["header"]["metadata_kv"][key][0])
            )
        for _ in range(gguf_file["header"]["tensor_count"]):
            name = f.read(int.from_bytes(f.read(8), "little")).decode("utf-8")
            gguf_file["tensor_infos"][name] = {
                "dimensions": [
                    int.from_bytes(f.read(8), "little")
                    for _ in range(int.from_bytes(f.read(4), "little"))
                ],
                "type": int.from_bytes(f.read(4), "little"),
                "offset": int.from_bytes(f.read(8), "little"),
            }
            typ = gguf_file["tensor_infos"][name]["type"]
            if typ == 0:  # GGML_TYPE_F32
                gguf_file["tensor_infos"][name]["type"] = np.float32
            elif typ == 1:  # GGML_TYPE_F16
                gguf_file["tensor_infos"][name]["type"] = np.float16
            elif typ in range(2, 10):  # GGML_TYPE_Q4_0 to GGML_TYPE_Q8_1
                gguf_file["tensor_infos"][name]["type"] = np.uint8
            elif typ in range(10, 16):  # GGML_TYPE_Q2_K to GGML_TYPE_Q8_K
                gguf_file["tensor_infos"][name]["type"] = np.uint8  # This is a placeholder. Change if necessary.
            elif typ == 16:  # GGML_TYPE_I8
                gguf_file["tensor_infos"][name]["type"] = np.int8
            elif typ == 17:  # GGML_TYPE_I16
                gguf_file["tensor_infos"][name]["type"] = np.int16
            elif typ == 18:  # GGML_TYPE_I32
                gguf_file["tensor_infos"][name]["type"] = np.int32
            else:
                raise ValueError(f"Unsupported ggml_type: {typ}")
        alignment = gguf_file["header"]["metadata_kv"].get("general.alignment", 32)
        print(f.tell())
        f.seek(
            f.tell() + (alignment - f.tell() % alignment)
        )  # Skip padding
        # print("ALIGNMENT", gguf_file["header"]["metadata_kv"].get("general.alignment", 32))
        # print(f.read(1000))
        # exit()
        gguf_file["tensor_data"] = f.read()
    return gguf_file


if __name__ == "__main__":
    load_gguf("weights/llama-2-7b.Q4_0.gguf")
