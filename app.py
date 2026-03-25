import streamlit as st
import cv2
import numpy as np
import os
import tempfile

st.set_page_config(page_title="图像位平面压缩演示", layout="wide")
st.title("🗜️ 图像位平面压缩演示工具")
st.caption("支持二值码 / 格雷码位平面，支持无压缩 / RLE / 块编码 / JPEG")

DEFAULT_MEDIUM_JPEG_QUALITY = 60


# ==================== 工具函数 ====================
def convert_to_gray_code(img):
    return np.bitwise_xor(img, np.right_shift(img, 1))


def gray_code_to_binary(gray_img):
    binary = gray_img.copy()
    for shift in [1, 2, 4]:
        binary ^= (binary >> shift)
    return binary


def run_length_encoding_binary(bitplane):
    flat = bitplane.flatten()
    if len(flat) == 0:
        return 0
    compressed = []
    current = int(flat[0])
    count = 1
    for v in flat[1:]:
        v = int(v)
        if v == current and count < 127:
            count += 1
        else:
            encoded = (current << 7) | count
            compressed.append(encoded)
            current = v
            count = 1
    encoded = (current << 7) | count
    compressed.append(encoded)
    return len(compressed)


def block_encoding_binary(bitplane, block_h=4, block_w=4):
    h, w = bitplane.shape
    total_bits = 0
    for y in range(0, h, block_h):
        for x in range(0, w, block_w):
            block = bitplane[y:y + block_h, x:x + block_w]
            if block.shape != (block_h, block_w):
                padded = np.zeros((block_h, block_w), dtype=np.uint8)
                padded[:block.shape[0], :block.shape[1]] = block
                block = padded
            if np.all(block == 0):
                total_bits += 2
            elif np.all(block == 1):
                total_bits += 2
            else:
                total_bits += 2 + (block_h * block_w)
    return int(np.ceil(total_bits / 8.0))


def gray_image_bitplane_compression(img, method):
    sizes = []
    compressed = []
    for bit in range(8):
        bitplane = ((img >> bit) & 1).astype(np.uint8)
        original_size = bitplane.size / 8
        if method == "无压缩":
            compressed_size = original_size
        elif method == "游程编码 (RLE)":
            compressed_size = run_length_encoding_binary(bitplane)
        elif method == "块编码":
            compressed_size = block_encoding_binary(bitplane)
        else:
            compressed_size = original_size
        sizes.append(original_size)
        compressed.append(compressed_size)
    return sizes, compressed


def compress_gray_with_jpeg(gray_img, jpeg_quality=DEFAULT_MEDIUM_JPEG_QUALITY):
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]
    ok, encoded = cv2.imencode(".jpg", gray_img, encode_params)
    if not ok:
        raise RuntimeError("JPEG 编码失败")
    jpeg_bytes = encoded.tobytes()
    jpeg_size = len(jpeg_bytes)
    decoded = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    return decoded, jpeg_size


def reconstruct_from_remaining_bitplanes(img, discard):
    out = np.zeros_like(img, dtype=np.uint8)
    for bit in range(discard, 8):
        plane = ((img >> bit) & 1).astype(np.uint8)
        out = np.bitwise_or(out, np.left_shift(plane, bit))
    return out


# ==================== 上传图片 ====================
uploaded = st.file_uploader("📁 选择一张图片", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"])

if uploaded is not None:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = img_gray.shape
    original_size = h * w

    st.markdown("---")

    # ==================== 控制面板 ====================
    col_ctrl1, col_ctrl2, col_ctrl3, col_ctrl4 = st.columns(4)

    with col_ctrl1:
        mode = st.selectbox("位平面类型", ["二值码位平面", "格雷码位平面"])
    with col_ctrl2:
        method = st.selectbox("压缩方法", ["无压缩", "游程编码 (RLE)", "块编码", "JPEG"])
    with col_ctrl3:
        discard = st.selectbox("丢弃低位数 (0-8)", list(range(9)), index=0)
    with col_ctrl4:
        if method == "JPEG":
            jpeg_quality = st.slider("JPEG 质量", 1, 100, DEFAULT_MEDIUM_JPEG_QUALITY)
        else:
            jpeg_quality = DEFAULT_MEDIUM_JPEG_QUALITY
            st.markdown("&nbsp;")  # 占位

    st.markdown("---")

    # ==================== 计算 ====================
    display_img = img_gray.copy()

    if mode == "格雷码位平面":
        calc_img = convert_to_gray_code(img_gray)
    else:
        calc_img = img_gray.copy()

    if method == "JPEG":
        jpeg_decoded, compressed_bytes = compress_gray_with_jpeg(img_gray, jpeg_quality)
        bitplane_preview_img = jpeg_decoded.copy()
        display_img = jpeg_decoded.copy()
    else:
        bitplane_preview_img = calc_img.copy()
        sizes, compressed = gray_image_bitplane_compression(calc_img, method)
        total = 0
        for i in range(discard, 8):
            total += min(sizes[i], compressed[i])
        compressed_bytes = int(np.ceil(total))

    # 丢弃低位后重建
    display_img = reconstruct_from_remaining_bitplanes(display_img, discard)

    # 压缩率
    if compressed_bytes > 0:
        ratio = original_size / compressed_bytes
    else:
        ratio = 0

    # ==================== 信息展示 ====================
    info_cols = st.columns(4)
    info_cols[0].metric("图像尺寸", f"{w} × {h}")
    info_cols[1].metric("原始大小", f"{original_size:,} bytes")
    info_cols[2].metric("压缩后大小", f"{compressed_bytes:,} bytes")
    info_cols[3].metric("压缩率", f"{ratio:.2f}" if ratio > 0 else "N/A")

    st.markdown("---")

    # ==================== 原图 vs 压缩图 ====================
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("### 🖼️ 原始灰度图")
        st.image(img_gray, use_container_width=True, clamp=True)
    with col_right:
        st.markdown("### 🗜️ 压缩/丢弃后图像")
        st.image(display_img, use_container_width=True, clamp=True)

    st.markdown("---")

    # ==================== 8个位平面 ====================
    st.markdown("### 📊 位平面分解（8个位平面）")

    row1_cols = st.columns(4)
    row2_cols = st.columns(4)
    all_bp_cols = row1_cols + row2_cols

    for i in range(8):
        with all_bp_cols[i]:
            bitplane = ((bitplane_preview_img >> i) & 1) * 255
            st.markdown(f"**Bit {i}**（{'高位' if i >= 4 else '低位'}）")
            st.image(bitplane.astype(np.uint8), use_container_width=True, clamp=True)

else:
    st.info("👆 请先上传一张图片，然后选择位平面类型和压缩方法进行演示。")
