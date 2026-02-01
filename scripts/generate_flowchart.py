"""
???????????????

?????
- ???#02304A(??), #136783(??), #219EBC(??), #90C9E7(??)
- ?????DAWS? #C9A24D
- ????/?/?
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Polygon
import numpy as np
from pathlib import Path
import shutil

# 严格蓝灰配色
PALETTE = {
    "navy": "#02304A",         # 藏青 - 核心/锚点
    "deep_blue": "#136783",    # 深蓝 - 分支
    "cyan_blue": "#219EBC",    # 中蓝 - 时代
    "light_blue": "#90C9E7",   # 浅蓝 - 结果节点
    "arrow_gray": "#4A5A66",   # 箭头灰蓝
    "text_gray": "#4A5A66",    # 文字灰蓝（同arrow_gray）
    "white": "#FFFFFF",
    "daws_gold": "#C9A24D",    # 唯一强调色（仅DAWS）
}


def draw_box(
    ax,
    x,
    y,
    w,
    h,
    text,
    facecolor,
    textcolor='white',
    fontsize=11,
    edgecolor=None,
    linestyle='-',
    linewidth=0,
    alpha=1.0,
    fontweight='bold',
    text_kwargs=None,
):
    """绘制圆角矩形"""
    box = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        facecolor=facecolor,
        edgecolor=edgecolor if edgecolor else facecolor,
        linewidth=linewidth, linestyle=linestyle,
        transform=ax.transData,
        alpha=alpha,
    )
    ax.add_patch(box)
    text_props = dict(
        ha='center',
        va='center',
        fontsize=fontsize,
        fontweight=fontweight,
        color=textcolor,
        wrap=True,
    )
    if text_kwargs:
        text_props.update(text_kwargs)

    ax.text(
        x,
        y,
        text,
        **text_props,
    )


def draw_arrow(ax, start, end, color, lw=2, alpha=1.0, connectionstyle=None):
    """绘制箭头"""
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(
                    arrowstyle='->',
                    color=color,
                    lw=lw,
                    alpha=alpha,
                    mutation_scale=12,
                    connectionstyle=connectionstyle,
                ))


def draw_diamond(ax, x, y, size, text, facecolor, fontsize=10):
    """绘制菱形"""
    diamond = Polygon(
        [(x, y+size), (x+size*0.85, y), (x, y-size), (x-size*0.85, y)],
        facecolor=facecolor, edgecolor=facecolor, linewidth=0
    )
    ax.add_patch(diamond)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            fontweight='bold', color='white')






def generate_show_flowchart():
    """
    Figure 1 - Audit links judges, fans, and eliminations across eras

    ???
    - Weekly Performance/Combined Score: #02304A??
    - Judges/Fans/Era??: #136783??
    - Percent/Rank Era: #219EBC??
    - Bottom 2??: #02304A??
    - ????: ??#90C9E7??????
      - Safe: ???
      - Saved: ????
      - Eliminated: ??????
    """
    fig, ax = plt.subplots(figsize=(12.8, 7.6))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6.8)
    ax.set_aspect('equal')
    ax.axis('off')

    font_bump = 1

    # ??
    ax.text(
        5.5,
        6.1,
        "Audit links judges, fans, and eliminations across eras",
        ha='center',
        fontsize=17 + font_bump,
        fontweight='bold',
        color=PALETTE["navy"],
    )

    # === ???????? ===
    weekly_box = dict(x=5.5, y=5.25, w=2.0, h=0.8)
    draw_box(
        ax,
        weekly_box["x"],
        weekly_box["y"],
        weekly_box["w"],
        weekly_box["h"],
        "Weekly\nPerformance",
        PALETTE["navy"],
        fontsize=13 + font_bump,
    )
    combined_box = dict(x=5.5, y=3.30, w=2.85, h=0.8)
    draw_box(
        ax,
        combined_box["x"],
        combined_box["y"],
        combined_box["w"],
        combined_box["h"],
        "Combined Score\n$\\alpha \\cdot \\mathrm{Judge} + (1-\\alpha)\\cdot \\mathrm{Fan}$",
        PALETTE["navy"],
        fontsize=12 + font_bump,
        text_kwargs={
            "clip_on": False,
            "bbox": dict(boxstyle="square,pad=0.45", facecolor="none", edgecolor="none"),
        },
    )
    ax.text(
        7.25,
        2.68,
        "Audit fusion",
        ha='left',
        fontsize=8 + font_bump,
        color=PALETTE["text_gray"],
        alpha=0.8,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.6, pad=0.15),
    )

    # === ?????? ===
    judges_box = dict(x=4.15, y=4.27, w=1.35, h=0.52)
    draw_box(
        ax,
        judges_box["x"],
        judges_box["y"],
        judges_box["w"],
        judges_box["h"],
        "Judges\n(4 experts)",
        PALETTE["deep_blue"],
        fontsize=9 + font_bump,
        alpha=0.85,
        fontweight='semibold',
    )
    fans_box = dict(x=6.85, y=4.27, w=1.35, h=0.52)
    draw_box(
        ax,
        fans_box["x"],
        fans_box["y"],
        fans_box["w"],
        fans_box["h"],
        "Fans\n(Public Vote)",
        PALETTE["deep_blue"],
        fontsize=9 + font_bump,
        alpha=0.85,
        fontweight='semibold',
    )

    # Scoring Era ??
    ax.text(
        5.5,
        2.20,
        "Scoring Era",
        ha='center',
        fontsize=12 + font_bump,
        fontweight='bold',
        color=PALETTE["deep_blue"],
    )

    # === ?????? ===
    percent_box = dict(x=4.15, y=1.85, w=1.5, h=0.46)
    draw_box(
        ax,
        percent_box["x"],
        percent_box["y"],
        percent_box["w"],
        percent_box["h"],
        "Percent Era\n(S3-27)",
        PALETTE["light_blue"],
        textcolor=PALETTE["navy"],
        fontsize=9 + font_bump,
        edgecolor=PALETTE["deep_blue"],
        linestyle='-',
        linewidth=1.1,
        alpha=0.85,
        fontweight='semibold',
    )
    rank_box = dict(x=6.4, y=1.85, w=1.5, h=0.46)
    draw_box(
        ax,
        rank_box["x"],
        rank_box["y"],
        rank_box["w"],
        rank_box["h"],
        "Rank Era\n(S1-2, S28+)",
        PALETTE["light_blue"],
        textcolor=PALETTE["navy"],
        fontsize=9 + font_bump,
        edgecolor=PALETTE["deep_blue"],
        linestyle='--',
        linewidth=1.1,
        alpha=0.85,
        fontweight='semibold',
    )
    ax.annotate(
        '',
        xy=(5.62, 1.85),
        xytext=(5.38, 1.85),
        arrowprops=dict(
            arrowstyle='->',
            color=PALETTE["arrow_gray"],
            lw=1.2,
            linestyle='--',
            alpha=0.8,
            mutation_scale=12,
        ),
    )
    ax.text(5.5, 1.93, "rule shift", ha='center', fontsize=8 + font_bump, color=PALETTE["text_gray"], alpha=0.75)

    # === Bottom 2 + outcomes ===
    diamond_center = dict(x=5.5, y=1.09, size=0.5)
    draw_diamond(
        ax,
        diamond_center["x"],
        diamond_center["y"],
        diamond_center["size"],
        "Bottom\n2",
        PALETTE["navy"],
        fontsize=10 + font_bump,
    )
    safe_box = dict(x=3.55, y=0.48, w=1.6, h=0.46)
    draw_box(
        ax,
        safe_box["x"],
        safe_box["y"],
        safe_box["w"],
        safe_box["h"],
        "Safe",
        PALETTE["light_blue"],
        textcolor=PALETTE["navy"],
        fontsize=9 + font_bump,
        alpha=0.85,
        fontweight='semibold',
    )
    saved_box = dict(x=5.5, y=0.48, w=1.6, h=0.46)
    draw_box(
        ax,
        saved_box["x"],
        saved_box["y"],
        saved_box["w"],
        saved_box["h"],
        "Saved",
        PALETTE["light_blue"],
        textcolor=PALETTE["navy"],
        fontsize=9 + font_bump,
        edgecolor=PALETTE["deep_blue"],
        linestyle='--',
        linewidth=1.1,
        alpha=0.85,
        fontweight='semibold',
    )
    eliminated_box = dict(x=7.45, y=0.48, w=1.6, h=0.46)
    draw_box(
        ax,
        eliminated_box["x"],
        eliminated_box["y"],
        eliminated_box["w"],
        eliminated_box["h"],
        "Eliminated",
        PALETTE["light_blue"],
        textcolor=PALETTE["navy"],
        fontsize=9 + font_bump,
        edgecolor=PALETTE["deep_blue"],
        linestyle='-',
        linewidth=2.0,
        alpha=0.85,
        fontweight='semibold',
    )

    # === ????????? ===
    def box_anchor(box, side):
        if side == "top":
            return (box["x"], box["y"] + box["h"] / 2)
        if side == "bottom":
            return (box["x"], box["y"] - box["h"] / 2)
        if side == "left":
            return (box["x"] - box["w"] / 2, box["y"])
        if side == "right":
            return (box["x"] + box["w"] / 2, box["y"])
        raise ValueError(f"Unknown side: {side}")

    def diamond_anchor(center, side):
        if side == "top":
            return (center["x"], center["y"] + center["size"])
        if side == "bottom":
            return (center["x"], center["y"] - center["size"])
        if side == "left":
            return (center["x"] - 0.85 * center["size"], center["y"])
        if side == "right":
            return (center["x"] + 0.85 * center["size"], center["y"])
        raise ValueError(f"Unknown side: {side}")

    ac = PALETTE["arrow_gray"]
    lw_main = 2.2
    lw_secondary = 2.0
    lw_annot = 1.2

    draw_arrow(ax, box_anchor(weekly_box, "bottom"), box_anchor(judges_box, "top"), ac, lw=lw_main)
    draw_arrow(ax, box_anchor(weekly_box, "bottom"), box_anchor(fans_box, "top"), ac, lw=lw_main)
    draw_arrow(
        ax,
        box_anchor(judges_box, "bottom"),
        (combined_box["x"] - combined_box["w"] * 0.25, combined_box["y"] + combined_box["h"] / 2),
        ac,
        lw=lw_main,
        connectionstyle="arc3,rad=0.15",
    )
    draw_arrow(
        ax,
        box_anchor(fans_box, "bottom"),
        (combined_box["x"] + combined_box["w"] * 0.25, combined_box["y"] + combined_box["h"] / 2),
        ac,
        lw=lw_main,
        connectionstyle="arc3,rad=-0.15",
    )
    draw_arrow(ax, box_anchor(combined_box, "bottom"), (combined_box["x"], 2.52), ac, lw=lw_main)
    draw_arrow(ax, (5.15, 2.12), box_anchor(percent_box, "top"), ac, lw=lw_main)
    draw_arrow(ax, (5.85, 2.12), box_anchor(rank_box, "top"), ac, lw=lw_main)
    draw_arrow(ax, box_anchor(percent_box, "bottom"), (diamond_center["x"] - 0.25, diamond_center["y"] + diamond_center["size"]), ac, lw=lw_secondary)
    draw_arrow(ax, box_anchor(rank_box, "bottom"), (diamond_center["x"] + 0.25, diamond_center["y"] + diamond_center["size"]), ac, lw=lw_secondary)
    draw_arrow(ax, diamond_anchor(diamond_center, "left"), box_anchor(safe_box, "top"), ac, lw=lw_main)
    draw_arrow(ax, diamond_anchor(diamond_center, "bottom"), box_anchor(saved_box, "top"), ac, lw=lw_main)
    draw_arrow(ax, diamond_anchor(diamond_center, "right"), box_anchor(eliminated_box, "top"), ac, lw=lw_main)
    # === ?? ===
    legend_elements = [
        mpatches.Patch(facecolor=PALETTE["light_blue"], edgecolor=PALETTE["light_blue"], label='Safe'),
        mpatches.Patch(facecolor=PALETTE["light_blue"], edgecolor=PALETTE["deep_blue"],
                       linestyle='--', linewidth=1.5, label='Saved (dashed)'),
        mpatches.Patch(facecolor=PALETTE["light_blue"], edgecolor=PALETTE["deep_blue"],
                       linewidth=2, label='Eliminated (bold)'),
    ]
    ax.legend(
        handles=legend_elements,
        loc='lower left',
        fontsize=9 + font_bump,
        frameon=True,
        framealpha=0.6,
        edgecolor=PALETTE["light_blue"],
    )

    # ??
    output_dir = Path(__file__).parent.parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / "fig_dwts_show_process.pdf"
    plt.tight_layout()
    plt.savefig(fig_path, bbox_inches='tight', dpi=300)
    plt.savefig(str(fig_path).replace('.pdf', '.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Fig 1 saved: {fig_path}")
    return fig_path


def generate_analytical_framework():
    """
    Figure 3 - Dual-Core Engine Architecture

    ???
    - Inversion Engine??: #02304A
    - LP/MILP Core: #136783
    - Bayesian/Downstream: #219EBC
    - ??: #4A5A66
    """
    fig, ax = plt.subplots(figsize=(13, 6.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.8)
    ax.set_aspect('equal')
    ax.axis('off')

    # ??
    ax.text(6, 5.7, "Uncertainty propagates through a dual-core engine", ha='center',
            fontsize=17, fontweight='bold', color=PALETTE["navy"])

    # === ??????? ===
    draw_box(ax, 1.5, 3.8, 2.0, 0.9, "Observed Data\n- Judge Scores\n- Eliminations",
             PALETTE["navy"], fontsize=10)

    # === ?????????? ===
    engine_box = FancyBboxPatch(
        (3.0, 1.3), 4.2, 4.0,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        facecolor='white', edgecolor=PALETTE["navy"], linewidth=2.5,
        alpha=0.95, transform=ax.transData
    )
    ax.add_patch(engine_box)
    ax.text(5.1, 5.0, "Inversion Engine", ha='center',
            fontsize=12, fontweight='bold', color=PALETTE["navy"])

    # LP/MILP Core????
    draw_box(ax, 4.1, 4.2, 1.6, 0.65, "LP Solver\n(Percent Era)",
             PALETTE["deep_blue"], fontsize=10)
    draw_box(ax, 6.1, 4.2, 1.6, 0.65, "MILP Solver\n(Rank Era)",
             PALETTE["light_blue"], textcolor=PALETTE["navy"], fontsize=10,
             edgecolor=PALETTE["deep_blue"], linewidth=1.2)

    # ????????
    draw_box(ax, 5.1, 3.1, 3.2, 0.55, "Feasible Intervals [L, U]",
             PALETTE["deep_blue"], fontsize=11)

    # Bayesian????
    draw_box(ax, 5.1, 2.0, 3.2, 0.55, "MaxEnt + Bayesian Sampling",
             PALETTE["cyan_blue"], textcolor=PALETTE["navy"],
             edgecolor=PALETTE["deep_blue"], linewidth=1.0, fontsize=10)

    # === ??????? ===
    draw_box(ax, 9.0, 4.5, 2.0, 0.6, "Posterior\nMean & HDI",
             PALETTE["cyan_blue"], textcolor=PALETTE["navy"],
             edgecolor=PALETTE["deep_blue"], linewidth=1.0, fontsize=10)
    draw_box(ax, 9.0, 3.6, 2.0, 0.5, "Counterfactual",
             PALETTE["cyan_blue"], textcolor=PALETTE["navy"],
             edgecolor=PALETTE["deep_blue"], linewidth=1.0, fontsize=10)
    draw_box(ax, 9.0, 2.9, 2.0, 0.5, "XGBoost + SHAP",
             PALETTE["cyan_blue"], textcolor=PALETTE["navy"],
             edgecolor=PALETTE["deep_blue"], linewidth=1.0, fontsize=10)
    draw_box(ax, 9.0, 2.2, 2.0, 0.5, "DAWS Mechanism",
             PALETTE["cyan_blue"], textcolor=PALETTE["navy"],
             edgecolor=PALETTE["deep_blue"], linewidth=1.0, fontsize=10)
    ax.text(7.75, 2.55, "Uncertainty\nPropagation", ha='center', fontsize=9, color=PALETTE["text_gray"])

    # === ???????? ===
    draw_box(ax, 11, 3.3, 0.9, 2.0, "Policy\nRecom.",
             PALETTE["navy"], fontsize=11)

    # === ?????? ===
    ac = PALETTE["arrow_gray"]
    draw_arrow(ax, (2.5, 3.8), (3.0, 3.8), ac, lw=2.4)
    draw_arrow(ax, (4.1, 3.85), (4.1, 3.4), ac, lw=2.4)
    draw_arrow(ax, (6.1, 3.85), (6.1, 3.4), ac, lw=2.4)
    draw_arrow(ax, (5.1, 2.8), (5.1, 2.3), ac, lw=2.4)
    draw_arrow(ax, (7.2, 4.2), (8.0, 4.5), ac, lw=2.4)
    draw_arrow(ax, (6.7, 2.0), (8.0, 3.6), ac, lw=2.4)
    draw_arrow(ax, (6.7, 2.0), (8.0, 2.9), ac, lw=2.4)
    draw_arrow(ax, (6.7, 2.0), (8.0, 2.2), ac, lw=2.4)
    draw_arrow(ax, (10.0, 3.6), (10.55, 3.5), ac, lw=2.2)
    draw_arrow(ax, (10.0, 2.9), (10.55, 3.2), ac, lw=2.2)
    draw_arrow(ax, (10.0, 2.2), (10.55, 2.8), ac, lw=2.2)

    # ??
    legend_elements = [
        mpatches.Patch(facecolor=PALETTE["deep_blue"], label='Core Solvers'),
        mpatches.Patch(facecolor=PALETTE["cyan_blue"], label='Downstream Analysis'),
        mpatches.Patch(facecolor=PALETTE["navy"], label='I/O Modules'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=11,
              frameon=True, framealpha=0.9, ncol=3, handlelength=1.8, handleheight=1.2, borderpad=0.6, columnspacing=1.2)

    # ??
    output_dir = Path(__file__).parent.parent / "figures"
    fig_path = output_dir / "fig_dwts_flowchart_vector.pdf"
    plt.tight_layout()
    plt.savefig(fig_path, bbox_inches='tight', dpi=300)
    plt.savefig(str(fig_path).replace('.pdf', '.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Fig 3 saved: {fig_path}")
    return fig_path
def main():
    show_fig = generate_show_flowchart()
    
    # 同步到论文目录
    paper_dirs = [
        Path(__file__).parent.parent / "paper" / "en" / "PaperC" / "figures",
        Path(__file__).parent.parent / "paper" / "zh" / "PaperC - Chinese" / "figures",
    ]
    for paper_dir in paper_dirs:
        paper_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(show_fig, paper_dir / "fig_dwts_show_process.pdf")
        print(f"Copied to: {paper_dir}")


if __name__ == "__main__":
    main()
