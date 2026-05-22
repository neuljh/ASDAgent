import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import font_manager, ft2font
from pathlib import Path


def configure_chinese_font():
    """Pick an available CJK font so saved figures render Chinese correctly."""
    local_font_dir = Path(__file__).with_name("fonts")
    sample_text = "中文"
    candidate_font_paths = []

    def supports_chinese(font_path):
        try:
            charmap = ft2font.FT2Font(str(font_path)).get_charmap()
        except Exception:
            return False
        return all(ord(char) in charmap for char in sample_text)

    env_font = os.environ.get("MATPLOTLIB_FONT")
    if env_font:
        candidate_font_paths.append(Path(env_font).expanduser())

    candidate_font_paths.extend(
        [
            local_font_dir / "NotoSansCJKsc-Regular.otf",
            local_font_dir / "SourceHanSansSC-Regular.otf",
            local_font_dir / "SimHei.ttf",
            Path.home() / ".local/share/fonts/NotoSansCJKsc-Regular.otf",
            Path.home() / ".local/share/fonts/SourceHanSansSC-Regular.otf",
            Path.home() / ".fonts/NotoSansCJKsc-Regular.otf",
            Path.home() / ".fonts/SourceHanSansSC-Regular.otf",
            Path.home() / "fonts/STKAITI.TTF",
            Path.home() / "fonts/SimHei.ttf",
            Path.home() / "fonts/NotoSansCJKsc-Regular.otf",
            Path.home() / "fonts/SourceHanSansSC-Regular.otf",
            Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
            Path("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"),
            Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
            Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),
            Path("/System/Library/Fonts/PingFang.ttc"),
            Path("/System/Library/Fonts/Hiragino Sans GB.ttc"),
            Path("C:/Windows/Fonts/msyh.ttc"),
            Path("C:/Windows/Fonts/simhei.ttf"),
        ]
    )

    for font_dir in [
        local_font_dir,
        Path.home() / ".local/share/fonts",
        Path.home() / ".fonts",
        Path.home() / "fonts",
    ]:
        if font_dir.is_dir():
            for pattern in ("*.otf", "*.ttf", "*.ttc"):
                candidate_font_paths.extend(sorted(font_dir.glob(pattern)))

    seen_paths = set()
    for font_path in candidate_font_paths:
        if font_path in seen_paths:
            continue
        seen_paths.add(font_path)
        if font_path.exists() and supports_chinese(font_path):
            font_manager.fontManager.addfont(str(font_path))
            return font_manager.FontProperties(fname=str(font_path)).get_name()

    candidate_font_names = [
        "Noto Sans CJK SC",
        "Noto Sans CJK TC",
        "Noto Sans CJK JP",
        "Source Han Sans SC",
        "Source Han Sans CN",
        "WenQuanYi Zen Hei",
        "WenQuanYi Micro Hei",
        "Microsoft YaHei",
        "SimHei",
        "PingFang SC",
        "Heiti SC",
        "Arial Unicode MS",
        "Sarasa Gothic SC",
        "STKaiti",
        "华文楷体",
    ]
    for font in font_manager.fontManager.ttflist:
        if font.name in candidate_font_names and supports_chinese(font.fname):
            return font.name

    raise RuntimeError(
        "未找到可用的中文字体。请安装 Noto Sans CJK SC / WenQuanYi Zen Hei，"
        "或将字体文件放到 ~/fonts，或将字体文件路径写入环境变量 MATPLOTLIB_FONT。"
    )


# 1. 设置中文字体（非常重要，否则中文会显示为乱码或方块）
font_name = configure_chinese_font()
plt.rcParams["font.family"] = [font_name, "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False  # 确保正常显示负号


def reciprocal(values):
    """Use reciprocal values for display so smaller divergence becomes larger score."""
    values = np.asarray(values, dtype=float)
    if np.any(values == 0):
        raise ValueError("散度值中包含 0，无法取倒数绘图。")
    return (1.0 / values).tolist()


def parse_args():
    parser = argparse.ArgumentParser(description="Plot reciprocal divergence radar chart.")
    parser.add_argument(
        "--method",
        choices=["offset", "log"],
        default="offset",
        help="offset: linear offset display without center circle; log: logarithmic radial axis.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/figures/doctor_child_divergence_radar_chart.png"),
        help="Path for the saved figure.",
    )
    return parser.parse_args()


def format_tick_label(value):
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def configure_radial_axis(ax, data_matrix, method):
    radial_min = float(data_matrix.min())
    radial_max = float(data_matrix.max())
    tick_values = np.linspace(radial_max / 5, radial_max, 5)

    if method == "offset":
        # Shift displayed radii outward instead of moving the true origin,
        # so the chart stays visually full without creating a center hole.
        radial_offset = radial_max * 0.22
        plot_data = data_matrix + radial_offset
        ax.set_ylim(0, radial_max * 1.05 + radial_offset)
        ax.set_yticks(tick_values + radial_offset)
        ax.set_yticklabels([format_tick_label(tick) for tick in tick_values], fontsize=10)
        title_suffix = ""
    else:
        plot_data = data_matrix
        log_ticks = np.geomspace(radial_min, radial_max, 5)
        ax.set_yscale("log")
        ax.set_ylim(radial_min * 0.9, radial_max * 1.05)
        ax.set_yticks(log_ticks)
        ax.set_yticklabels([format_tick_label(tick) for tick in log_ticks], fontsize=10)
        title_suffix = "（对数径向坐标）"

    ax.set_rlabel_position(22.5)
    return plot_data, title_suffix


def main():
    args = parse_args()

    # 2. 准备数据
    labels = np.array(['医生策略 - 1 / KL散度', '医生策略 - 1 / JS散度', '儿童反应 - 1 / KL散度', '儿童反应 - 1 / JS散度'])

    data_doc_child = [0.083, 0.019, 0.007, 0.002]  # DoctorAgent + ChildAgent
    data_doc_gpt = [0.325, 0.072, 0.039, 0.009]    # DoctorAgent + GPT-4o
    data_gpt_child = [0.259, 0.118, 0.024, 0.006]  # GPT-4o + ChildAgent

    raw_data = [data_doc_child, data_doc_gpt, data_gpt_child]
    reciprocal_data = np.asarray([reciprocal(values) for values in raw_data], dtype=float)
    model_names = [
        '医生Agent - 儿童Agent',
        '医生Agent - GPT-4o',
        'GPT-4o - 儿童Agent'
    ]

    # 3. 计算雷达图的角度
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    closed_angles = angles + angles[:1]

    # 4. 开始绘图
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_frame_on(False)

    plot_data, title_suffix = configure_radial_axis(ax=ax, data_matrix=reciprocal_data, method=args.method)
    closed_data = [row.tolist() + [row[0]] for row in plot_data]

    # 定义美观的配色方案（采用学术常用的高对比度颜色）
    colors = ['#2ca02c', '#1f77b4', '#ff7f0e'] # 绿色, 蓝色, 橙色

    # 循环绘制每一组数据
    for i in range(len(closed_data)):
        ax.plot(closed_angles, closed_data[i], color=colors[i], linewidth=2.5, label=model_names[i])
        ax.fill(closed_angles, closed_data[i], color=colors[i], alpha=0.15)

    # 5. 美化图表细节
    ax.grid(color='#d3d3d3', linestyle='--', linewidth=1, alpha=0.8)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=12, fontweight='bold')
    ax.spines['polar'].set_visible(False)

    plt.title(f'医生策略与儿童反应与真实分布的 KL 和 JS 散度倒数{title_suffix}', size=15, fontweight='bold', y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=11, frameon=True, shadow=True)
    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    main()
