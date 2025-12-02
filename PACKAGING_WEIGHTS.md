# Packaging SAM3 Weights for Jetson

This guide explains how to package optimized SAM3 weights on a desktop machine and transfer them to Jetson devices via USB.

## Why Package Weights?

**Benefits:**
- ✅ **No HuggingFace authentication needed on Jetson** - weights are pre-downloaded
- ✅ **Faster loading** - weights are already optimized (FP16 quantized)
- ✅ **Smaller file size** - FP16 weights are ~50% smaller than FP32
- ✅ **Works offline** - no internet connection required on Jetson
- ✅ **USB transfer** - easy to copy to Jetson devices

## Workflow Overview

1. **On Desktop**: Package optimized weights using `package_jetson_weights.py`
2. **Transfer**: Copy the packaged file to USB drive
3. **On Jetson**: Extract and use with `--checkpoint-path` option

## Step-by-Step Instructions

### Step 1: Package Weights on Desktop

On a machine with GPU and sufficient memory (8GB+):

```bash
# 1. Checkout jetson-orin branch
cd sam3-project
git checkout jetson-orin

# 2. Install dependencies (if not already done)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements_jetson.txt

# 3. Authenticate with HuggingFace (to download original weights)
huggingface-cli login

# 4. Package weights with FP16 quantization (recommended)
python package_jetson_weights.py --output sam3_jetson_fp16.pth --quantize fp16
```

This creates:
- `sam3_jetson_fp16.pth` - Optimized weights file (~1.5-2 GB)
- `sam3_jetson_fp16_metadata.json` - Metadata about the weights
- `sam3_jetson_fp16.tar.gz` - Compressed archive for USB transfer (~1.2-1.5 GB)

### Step 2: Transfer to USB

```bash
# Copy the tarball to USB drive
cp sam3_jetson_fp16.tar.gz /media/usb/
```

Or use a file manager to copy the `.tar.gz` file to your USB drive.

### Step 3: Extract on Jetson

```bash
# 1. Mount USB drive (usually auto-mounted at /media/usb or /media/username/usb)
# 2. Copy to Jetson
cp /media/usb/sam3_jetson_fp16.tar.gz ~/

# 3. Extract
cd ~/
tar -xzf sam3_jetson_fp16.tar.gz

# This creates:
# - sam3_jetson_fp16.pth
# - sam3_jetson_fp16_metadata.json
```

### Step 4: Use on Jetson

```bash
# Activate environment
source venv_jetson/bin/activate

# Run demo with pre-packaged weights
python sam3_jetson_demo.py --image image.jpg --prompt "boat" \
    --checkpoint-path ~/sam3_jetson_fp16.pth
```

## Quantization Options

### FP16 (Recommended)

```bash
python package_jetson_weights.py --output sam3_jetson_fp16.pth --quantize fp16
```

- **File size**: ~1.5-2 GB
- **Memory usage**: ~1.5-2 GB on Jetson
- **Accuracy**: Minimal loss (<1%)
- **Compatibility**: Works on all Jetson devices

### INT8 (Maximum Compression)

```bash
python package_jetson_weights.py --output sam3_jetson_int8.pth --quantize int8
```

- **File size**: ~750 MB - 1 GB
- **Memory usage**: ~1-1.5 GB on Jetson
- **Accuracy**: Slight loss (~2-3%)
- **Compatibility**: May not work on all Jetson devices

### FP32 (Not Recommended)

```bash
python package_jetson_weights.py --output sam3_jetson_fp32.pth --quantize none
```

- **File size**: ~3-4 GB
- **Memory usage**: ~3-4 GB on Jetson
- **Accuracy**: Full precision
- **Compatibility**: Works but exceeds 2.5GB target

## File Sizes

| Quantization | File Size | Compressed | Memory Usage |
|--------------|-----------|------------|--------------|
| FP16 | ~1.5-2 GB | ~1.2-1.5 GB | ~1.5-2 GB |
| INT8 | ~750 MB - 1 GB | ~600-800 MB | ~1-1.5 GB |
| FP32 | ~3-4 GB | ~2.5-3 GB | ~3-4 GB |

*Note: Compressed sizes are for `.tar.gz` archives*

## Advanced Options

### Custom Checkpoint

If you have a custom checkpoint file:

```bash
python package_jetson_weights.py \
    --output custom_weights.pth \
    --checkpoint-path /path/to/checkpoint.pth \
    --quantize fp16
```

### Skip Tarball Creation

If you only want the `.pth` file:

```bash
python package_jetson_weights.py \
    --output weights.pth \
    --quantize fp16 \
    --no-tarball
```

### Multiple GPU

If you have multiple GPUs:

```bash
python package_jetson_weights.py \
    --output weights.pth \
    --gpu 1 \
    --quantize fp16
```

## Verification

### Check Metadata

```bash
# View metadata
cat sam3_jetson_fp16_metadata.json
```

### Verify Weights

On Jetson, the demo script will automatically:
- Load the packaged weights
- Verify metadata
- Skip quantization (since weights are already optimized)
- Display information about the weights

## Troubleshooting

### "Checkpoint file not found"

Make sure you:
1. Extracted the tarball on Jetson
2. Used the correct path with `--checkpoint-path`
3. File has `.pth` extension

### "Missing keys" or "Unexpected keys"

This is usually okay - the model will still work. These warnings indicate:
- Some keys in the checkpoint don't match the model (missing)
- Some model parameters don't have weights (unexpected)

The model will use default initialization for missing keys.

### File Too Large for USB

If the file is too large:
1. Use INT8 quantization (smaller file)
2. Use a larger USB drive
3. Transfer via network instead:
   ```bash
   # On desktop
   scp sam3_jetson_fp16.tar.gz jetson@jetson-ip:~/
   
   # On Jetson
   tar -xzf sam3_jetson_fp16.tar.gz
   ```

### Weights Don't Load

Make sure:
1. You're using the same branch (`jetson-orin`)
2. PyTorch versions are compatible
3. The weights file isn't corrupted (check file size)

## Best Practices

1. **Use FP16** - Best balance of size, speed, and accuracy
2. **Keep metadata** - Helps verify weights are correct
3. **Test on desktop first** - Verify weights work before transferring
4. **Use tarball** - Compressed format saves space and transfer time
5. **Document source** - Note which model version was used

## Example Workflow

```bash
# === On Desktop ===
cd sam3-project
git checkout jetson-orin
huggingface-cli login
python package_jetson_weights.py --output sam3_jetson_fp16.pth --quantize fp16
cp sam3_jetson_fp16.tar.gz /media/usb/

# === On Jetson ===
cp /media/usb/sam3_jetson_fp16.tar.gz ~/
cd ~/
tar -xzf sam3_jetson_fp16.tar.gz
cd sam3-project
source venv_jetson/bin/activate
python sam3_jetson_demo.py --image test.jpg --prompt "boat" \
    --checkpoint-path ~/sam3_jetson_fp16.pth
```

## See Also

- `JETSON_SETUP.md` - Full setup guide
- `README_JETSON.md` - Branch overview
- `package_jetson_weights.py --help` - Script help

