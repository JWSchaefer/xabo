def style_axes(*axs):
    for ax in axs:
        for spine in ax.spines:
            ax.spines[spine].set_visible(False)
        ax.grid(True, which="major", axis="both", alpha=0.3)
